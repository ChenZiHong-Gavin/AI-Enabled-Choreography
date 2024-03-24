import torch
from functions.model import LSTM_VAE
from functions.load_data import MarielDataset
from functions.plotting import animate_stick
from tqdm import tqdm
import easydict
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)

writer = SummaryWriter(log_dir='../logs')

def train(args, model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epochs = tqdm(range(args.max_iter // len(train_loader) + 1))
    print("epochs", len(epochs))

    # training loop
    count = 0
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )

        for i, (seq, target) in train_iterator:
            if count >= args.max_iter:
                break
            count += 1

            seq, target = seq.to(model.device), target.to(model.device)
            past_data = seq.view(seq.size(0), seq.size(1), -1).float().to(model.device)
            target_data = target.view(target.size(0), target.size(1), -1).float().to(model.device)
            mloss, recon_x, info = model(past_data, target_data)

            writer.add_scalar('Loss/train', mloss.mean(), count)

            # Backward and optimize
            optimizer.zero_grad()
            mloss.mean().backward()
            optimizer.step()

            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], count)

            train_iterator.set_postfix({"loss": float(mloss.mean())})

        model.eval()
        eval_loss = 0
        test_iterator = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="testing"
        )

        with torch.no_grad():
            for i, (seq, target) in test_iterator:
                seq, target = seq.to(model.device), target.to(model.device)

                # reshape
                past_data = seq.view(seq.size(0), seq.size(1), -1).float().to(model.device)
                target_data = target.view(target.size(0), target.size(1), -1).float().to(model.device)

                mloss, recon_x, info = model(past_data, target_data)
                eval_loss += mloss.mean().item()

                test_iterator.set_postfix({"loss": float(mloss.mean())})

        eval_loss /= len(test_loader)
        print(f"Epoch {epoch}, Test Loss: {eval_loss}")

    return model

args = easydict.EasyDict({
        "batch_size": 1024,
        "input_size": 318,
        "hidden_size": 256,
        "latent_size": 128,
        "learning_rate": 0.00005,
        "num_layers": 1,
        "max_iter": 200,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "shuffle": False,
        "ouput_path": "../models/model.pth"
})

batch_size = args.batch_size
input_size = args.input_size
hidden_size = args.hidden_size
latent_size = args.latent_size
num_layers = args.num_layers

data = MarielDataset(reduced_joints=False, xy_centering=True, seq_len=128, predicted_timesteps=1, no_overlap=False, file_path="../data/mariel_*.npy")
train_indices = np.arange(int(0.9 * len(data)))
test_indices = np.arange(int(0.9 * len(data)), len(data))

train_set = torch.utils.data.Subset(data, train_indices)
test_set = torch.utils.data.Subset(data, test_indices)

dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=args.shuffle, drop_last=True)
dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=args.shuffle, drop_last=True)

def run_train():
    # define LSTM-based VAE model
    model = LSTM_VAE(input_size, hidden_size, latent_size, num_layers)
    model.to(args.device)
    train(args, model, dataloader_train, dataloader_test)
    torch.save(model.state_dict(), args.ouput_path)

def run_predict(index):
    past_data, _ = dataloader_test.dataset[index]
    past_data = torch.tensor(past_data).float()
    past_data = past_data.reshape(1, past_data.size(0), -1).float().to(args.device)

    with torch.no_grad():
        # load model
        model_to_load = LSTM_VAE(input_size, hidden_size, latent_size, num_layers)
        model_to_load.to(args.device)
        model_to_load.load_state_dict(torch.load(args.ouput_path))
        model_to_load.eval()

        recon_data = model_to_load(past_data, None)
        return recon_data

def run_generate(index, timesteps):
    past_data, _ = dataloader_test.dataset[index]
    past_data = torch.tensor(past_data).float()
    past_data = past_data.reshape(1, past_data.size(0), -1).float().to(args.device)

    with torch.no_grad():
        # load model
        model_to_load = LSTM_VAE(input_size, hidden_size, latent_size, num_layers)
        model_to_load.to(args.device)
        model_to_load.load_state_dict(torch.load(args.ouput_path))
        model_to_load.eval()

        for i in range(timesteps):
            recon_data = model_to_load(past_data[:, -128:, :], None)
            recon_data = recon_data.reshape(1, 1, 318)
            past_data = torch.cat((past_data, recon_data), dim=1)


        return past_data

if __name__ == "__main__":
    # run_train()

    # predict_frame = run_predict(1)
    # predict_frame = predict_frame[0].reshape(1, 53, 6).cpu().detach().numpy()[..., :3]
    # animate_stick(predict_frame)

    predict = run_generate(1, 50)
    predict = predict[0].reshape(predict.size(1), 53, 6).cpu().detach().numpy()[..., :3]
    print(predict.shape)
    animate_stick(predict)