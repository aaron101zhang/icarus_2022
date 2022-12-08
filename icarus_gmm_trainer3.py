print("importing icarus_mode_gmm")
from icarus_mode_gmm import ModeGPTLSTMGMM

print("icarus_mode_gmm imported")

import math
import time

import librosa
import numpy as np
import torchaudio.functional
from swda_gmm_dataset import SWDAGMMDataset

print("swda_gmm imported")
import os
import pandas
import argparse

from icarus_util import split_data
from icarus_silent_preds import translate_rms

import torch
import fairseq

# from meticulous import Experiment
print("Hello1")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
print("Hello")

params = {"w_size": 100, "f_shift": 50, "utt_len": 1200}

print("Initiating file system")
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(42)

"""
    Loading wav2vec model
"""
# cp_path = '/Users/zhanga67/Documents/School/NLP/icarus_2022/full_data_wav2vec/wav2vec_large.pt'

# cd drive/MyDrive/NLP/icarus
print("Starting wav2vec import")
cp_path = 'wav2vec/wav2vec_large.pt'

wav2vec_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
wav2vec_model = wav2vec_model[0]
wav2vec_model.to(device)
wav2vec_model.eval()
print("Finished wav2vec")
df = None

"""
    Param Sweep Functionality
"""
col_params = ['batch_size', 'lr']
# Template for each individual run
df_run = {}

st_ce = torch.nn.CrossEntropyLoss(reduction='none')


def calc_loss(output, target, ws):
    mu = output['mu']
    sigma = output['sigma']
    sigma = torch.maximum(sigma, 1e-5 * torch.ones(sigma.shape).to(device))
    h = output['h']
    A = h
    # print("Value A ===")
    # print(h.shape, target.shape, ws.shape)
    # print(target[0:10, 0:10], mu[0:10, 0:10])
    # print(mu.shape, sigma.shape)
    # print(ws[0:10])
    B = torch.exp(-0.5 * torch.square((target.unsqueeze(-1) - mu) / sigma)) * ( # batch_size x 60*train_size x No of Gaussians
                1 / (sigma * math.sqrt(2 * math.pi))) + 1e-4
    assert A.shape == B.shape
    S = torch.sum(A * B, dim=-1, keepdim=False)
    NLL = torch.mean(-torch.log(S) * ws)
    return NLL


def ce_da_loss(output, target, w_vec):
    output = torch.transpose(output, 1, 2)
    losses = st_ce(output, target) * w_vec
    return torch.mean(losses)


def pass_wav2vec(data):
    data = data.to(device)
    if len(data.shape) == 1:
        data = data.unsqueeze(0)
    batch_size = 32
    results = []
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            # print(min(i + batch_size, data.shape[0]))
            # print(data[i: min(i + batch_size, data.shape[0]), :].size())
            # print(data[i: min(i + batch_size, data.shape[0]), :])
            if (data[i: min(i + batch_size, data.shape[0]), :].size()[1] < 512):
                results.append(torch.ones((2, 512, 0)).to(device))
                continue
            z = wav2vec_model.feature_extractor(data[i: min(i + batch_size, data.shape[0]), :])
            c = wav2vec_model.feature_aggregator(z)
            # print(c)
            # print(c.size())
            del z
            results.append(c)
            torch.cuda.empty_cache()
    if len(results) > 1:
        results = torch.cat(results, dim=0)
    else:
        results = results[0]
    return results


def obtain_wav2vec(data):
    total = split_data(data)
    with torch.no_grad():
        wavs = [pass_wav2vec(w) for w in total]
    wav = torch.cat(wavs, dim=2)
    """
        Only obtain every 5 frames to maintain 50 ms step size
    """
    wav = torch.index_select(wav, 2, torch.tensor([5 * i for i in range(int(wav.shape[-1] / 5))]).to(device))
    wav = torch.transpose(wav, 1, 2)
    return wav


def get_wav2vec_rms(wav):
    rms = torch.tensor(translate_rms(wav)).unsqueeze(-1).to(device)
    wav2vecs = obtain_wav2vec(wav)
    samp_len = min(wav2vecs.shape[1], rms.shape[1])
    wav2vecs = wav2vecs[:, :samp_len, :]
    rms = rms[:, :samp_len, :]
    return torch.cat([wav2vecs, rms], dim=-1)


def get_acoustic_rms(wav):
    sr = 16000
    rms = torch.tensor(translate_rms(wav)).unsqueeze(-1).to(device)
    freqs_0 = librosa.yin((wav.numpy())[0], fmin=440, fmax=880, sr=16000,
                          frame_length=int((params["w_size"] / 1000) * sr),
                          hop_length=int((params['f_shift'] / 1000) * sr), center=False)
    freqs_1 = librosa.yin((wav.numpy())[1], fmin=440, fmax=880, sr=16000,
                          frame_length=int((params["w_size"] / 1000) * sr),
                          hop_length=int((params['f_shift'] / 1000) * sr), center=False)
    freqs = torch.tensor(np.vstack([freqs_0, freqs_1])).unsqueeze(-1)
    pitches = torchaudio.functional.compute_kaldi_pitch(wav, sample_rate=sr, frame_length=params["w_size"],
                                                        frame_shift=params['f_shift'])
    pitches = pitches[:, :, 0].unsqueeze(-1)
    return (torch.cat([freqs.to(device), pitches.to(device), rms.to(device)], dim=-1)).float()
    # if pitches.shape[1] != freqs.shape[1]:
    #     min_shape = min(pitches)


def train(model, train_loader, epoch, loss_decay, alpha):
    global df, col_params, add_std_loss, add_consistency_loss
    model.train()
    # Compute norms
    # Initiate running losses
    run_st_ts_loss = 0
    run_da_loss = 0
    run_std_loss = 0
    for id, (data, labels) in enumerate(train_loader):
        """
            Pass through the wav2vec model
        """
        optimizer.zero_grad()
        hidden = model.init_hidden(data.shape[0])
        labels['word_inds'] = labels['word_inds'].to(device)
        out = model(data, hidden, labels['transcript'], labels['word_inds'])
        
        ws = labels['gmm_w']
        
        # Reweight loss
        if loss_decay is not None:
            if loss_decay == 'linear':
                ws = labels['gmm_w']
                for i in range(ws.shape[0]):
                    found1=False
                    for j in range(1,ws.shape[1]):
                        if ws[i, j] == 1 and ws[i, j - 1] == 0:
                            cnt = 1
                            found1 = True
                        if (ws[i,j] == 0 and ws[i, j-1] == 1) or j == ws.shape[1]-1 :
                            ws[i,j-cnt:j] = 2*torch.FloatTensor(range(cnt))/cnt
                            found1 = False
                        if found1:
                            cnt += 1
            elif loss_decay == 'exp':
                alpha = alpha
                ws = labels['gmm_w']
                for i in range(ws.shape[0]):
                    found1=False
                    for j in range(1,ws.shape[1]):
                        if ws[i, j] == 1 and ws[i, j - 1] == 0:
                            cnt = 1
                            found1 = True
                        if (ws[i,j] == 0 and ws[i, j-1] == 1) or j == ws.shape[1]-1 :
                            expon = torch.exp(alpha*torch.FloatTensor(range(cnt))/cnt)
                            ws[i,j-cnt:j] = (expon/torch.sum(expon))*cnt
                            found1 = False
                        if found1:
                            cnt += 1
            elif loss_decay == 'log':
                alpha = alpha
                ws = labels['gmm_w']
                for i in range(ws.shape[0]):
                    found1=False
                    for j in range(1,ws.shape[1]):
                        if ws[i, j] == 1 and ws[i, j - 1] == 0:
                            cnt = 1
                            found1 = True
                        if (ws[i,j] == 0 and ws[i, j-1] == 1) or j == ws.shape[1]-1 :
                            logg = torch.log(1.001+alpha*torch.FloatTensor(range(cnt))/cnt)
                            ws[i,j-cnt:j] = (logg/torch.sum(logg))*cnt
                            #print(ws[i,j-cnt:j])
                            found1 = False
                        if found1:
                            cnt += 1
            elif loss_decay == 'sig':
                alpha = alpha
                ws = labels['gmm_w']
                for i in range(ws.shape[0]):
                    found1 = False
                    for j in range(1, ws.shape[1]):
                        if ws[i, j] == 1 and ws[i, j - 1] == 0:
                            cnt = 1
                            found1 = True
                        if (ws[i, j] == 0 and ws[i, j - 1] == 1) or j == ws.shape[1] - 1:
                            sigg = 1 / (1 + torch.exp(-(alpha / cnt) * (torch.FloatTensor(range(cnt)) - cnt / 2)))
                            ws[i, j - cnt:j] = (sigg / torch.sum(sigg)) * cnt
                            # print(ws[i,j-cnt:j])
                            found1 = False
                        if found1:
                            cnt += 1
                    
        #st_ts_loss = calc_loss(out, labels['start_ts'].to(device), labels['gmm_w'].to(device))
        st_ts_loss = calc_loss(out, labels['start_ts'].to(device), ws.to(device))
        run_st_ts_loss += st_ts_loss.detach().cpu().item()

        losses = st_ts_loss  # Removing end ts loss and d_act

        losses.backward(retain_graph=True)

        optimizer.step()

    run_st_ts_loss = run_st_ts_loss / len(train_loader)
    # Getting rid of end loss
    run_std_loss = run_std_loss / len(train_loader)
    run_da_loss = run_da_loss / len(train_loader)
    torch.cuda.empty_cache()

    # if id % 100 == 0:
    print("Start TS Loss = {l1}, DA Loss = {l2}".format(
        l1=run_st_ts_loss, l2=run_da_loss))
    return (run_st_ts_loss, {"st_loss_avg_" + str(epoch): run_st_ts_loss,
                             "std_loss_avg_" + str(epoch): run_std_loss
                             })


def test(model, test_loader, epoch):
    model.eval()
    st_ts = 0
    da = 0
    for test_data, labels in test_loader:
        # test_data = obtain_wav2vec(test_data)
        labels['word_inds'] = labels['word_inds'].to(device)
        out = model(test_data, model.init_hidden(test_data.shape[0]), labels['transcript'], labels['word_inds'])
        st_ts += calc_loss(out, labels['start_ts'].to(device), labels['gmm_w'].to(device)).item()
    st_ts = st_ts / len(test_loader)
    da = da / len(test_loader)
    print('Eval Losses: Start, Dialogue => ', st_ts, da)
    return st_ts, {"st_loss_eval_" + str(epoch): st_ts}


if __name__ == "__main__":
    print("Begin.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--message", type=str)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--drop_prob", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", type=int, default=3, help="0 if GPT2 + wav2vec, 1 if wav2vec only, 2 if GPT2 only, "
                                                            "3 if GPT2 + wav2vec + RMS, 4 if GPT2 + Acoustics")
    parser.add_argument("--dmax", type=int, default=2)
    parser.add_argument("--T", type=int, default=15)
    parser.add_argument("--adj", type=int, help="Adjustment mode for SWDA GMM Dataset", default=2)
    parser.add_argument("--save_dir", type=str, default="saved_weights_GMM")
    parser.add_argument("--weights_name", type=str, default="model-after-train.pt")  # assert file exists in save_dir,
    parser.add_argument("--use_small_set", type=bool, default=False)
    parser.add_argument("--recalc_wav2vec", type=int, default=0)
    parser.add_argument("--loss_decay", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    # Check args

    # Experiment.add_argument_group(parser)
    # experiment = Experiment.from_parser(parser)
    args = parser.parse_args()
    print("Args parsed.")
    # experiment.metadata.update({"Purpose Message": args.message})
    print({"Purpose Message": args.message})
    #curexpdir = os.path.join(os.path.dirname(os.getcwd()), args.save_dir)
    curexpdir = os.path.join(os.path.dirname(os.getcwd())+'/icarus', args.save_dir)
    print(curexpdir)
    if (os.path.isdir(curexpdir)):
        print("Directory already exists: Please choose a new name for this experiment")

    # cwd should always be NLP_Project/icarus, we move up to NLP_Project to store our outputs

    d_max = args.dmax

    if args.debug:
        size_limit = 3
    else:
        size_limit = -1

    if args.recalc_wav2vec:
        print(args.recalc_wav2vec)
        print('NOOO')
        if args.use_small_set:
            train_set = SWDAGMMDataset([], f_id="test_20", debug=args.debug, size_limit=size_limit, d_max=d_max,
                                       mode=args.adj)
        else:
            train_set = SWDAGMMDataset([], f_id="train_200", debug=args.debug, size_limit=size_limit, d_max=d_max,
                                       mode=args.adj)
        val_set = SWDAGMMDataset([], f_id="val_20", debug=args.debug, size_limit=size_limit, d_max=d_max, mode=args.adj)
        print("Datasets Gathered.")
        # test_set = SWDAGMMDataset([], f_id="test_20", debug=args.debug, size_limit=size_limit, d_max=d_max, mode=args.adj)
        """
            Use wav2vec to obtain embeddings on the fly
        """
        if args.mode != 2:
            for i in range(len(train_set.wavs)):
                if args.mode == 3:
                    print("training", i, len(train_set.wavs[i]))
                    train_set.wavs[i] = get_wav2vec_rms(train_set.wavs[i])
                elif args.mode == 4:
                    train_set.wavs[i] = get_acoustic_rms(train_set.wavs[i])
                else:
                    train_set.wavs[i] = obtain_wav2vec(train_set.wavs[i])
            for i in range(len(val_set.wavs)):
                if args.mode == 3:
                    print("val", i, len(val_set.wavs[i]))
                    val_set.wavs[i] = get_wav2vec_rms(val_set.wavs[i])
                elif args.mode == 4:
                    val_set.wavs[i] = get_acoustic_rms(val_set.wavs[i])
                else:
                    val_set.wavs[i] = obtain_wav2vec(val_set.wavs[i])
        print("wav2vec_model done")
        del wav2vec_model
        print("wav2vec_model done")

        # Save train and val sets
        torch.save(train_set, os.path.dirname(os.getcwd())+'/icarus/wav2vec/train_set')
        torch.save(val_set, os.path.dirname(os.getcwd())+'/icarus/wav2vec/val_set')

        torch.cuda.empty_cache()

    else:
        train_set = torch.load(os.path.dirname(os.getcwd())+'/icarus/wav2vec/train_set')
        val_set = torch.load(os.path.dirname(os.getcwd())+'/icarus/wav2vec/val_set')


    hyperparams = {'n_feature': train_set.get_feats_size()}
    kwargs = {'num_workers': 0, 'pin_memory': False} if device == 'cuda:0' else {}  # needed for using datasets on gpu

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               **kwargs)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                                             **kwargs)

    # Check if tsv file exists, if not create new file
    # File based on batch size
    df_run = {"lr": args.lr, "batch_size": args.batch_size}
    print(df_run)

    if args.model is None:
        if args.T is None:
            model = ModeGPTLSTMGMM(hyperparams['n_feature'], mode=args.mode, T=d_max * 2 * 4)
        else:
            model = ModeGPTLSTMGMM(hyperparams['n_feature'], mode=args.mode, T=args.T)
    else:
        model = torch.load(args.model)
    # model = model.to(device)
    model = model.cuda(device=0)
    model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    clip = 5  # gradient clipping

    best_loss = 10e3
    best_tr_loss = 10e3
    for epoch in range(1, args.epoch + 1):
        start_time = time.time()
        df = pandas.DataFrame(columns=col_params)
        print("Epoch {s} ============".format(s=epoch))
        # scheduler.step()
        tr_loss, tr_data = train(model, train_loader, epoch, args.loss_decay, args.alpha)
        ev_loss, eval_data = test(model, val_loader, epoch)
        df_run.update(tr_data)
        df_run.update(eval_data)
        if ev_loss < best_loss:
            best_loss = ev_loss
            torch.save(model, curexpdir + "/best-loss-model.pt")
            print({'eval_loss': ev_loss, 'current_epoch': epoch})
            # experiment.summary({'eval_loss': ev_loss, 'current_epoch': epoch})
        if tr_loss < best_tr_loss:
            best_tr_loss = tr_loss
            torch.save(model, curexpdir + "/train-loss-model.pt")
        if epoch == 4 or epoch == 9:
            torch.save(model, curexpdir + "/epoch-model-" + str(epoch) + ".pt")
        print("Epoch Took:", time.time() - start_time)
        if os.path.exists(curexpdir + "/run_stats.csv"):
            os.remove(curexpdir + "/run_stats.csv")
        df = df.append(df_run, ignore_index=True)
        df.to_csv(curexpdir + "/run_stats.csv", index=False)
    torch.save(model, curexpdir + "/" + args.weights_name)