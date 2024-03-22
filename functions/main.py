from model import LSTM_VAE
import torch
from torchvision import transforms
from tqdm import tqdm
from utils.MovingMNIST import MovingMNIST
import easydict
import matplotlib.pyplot as plt


def train(args, model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epochs = tqdm(range(args.max_iter // len(train_loader) + 1))

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

            # reshape
            past_data = seq.view(seq.size(0), seq.size(1), -1).float().to(model.device)

            # print(past_data.shape)
            mloss, recon_x, info = model(past_data)

            # Backward and optimize
            optimizer.zero_grad()
            mloss.mean().backward()
            optimizer.step()

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

                mloss, recon_x, info = model(past_data)
                eval_loss += mloss.mean().item()

                test_iterator.set_postfix({"loss": float(mloss.mean())})

        eval_loss /= len(test_loader)
        print(f"Epoch {epoch}, Test Loss: {eval_loss}")

    return model


def imshow(past_data, title="MovingMNIST"):
    num_img = len(past_data)
    fig = fig = plt.figure(figsize=(4 * num_img, 4))

    for idx in range(1, num_img + 1):
        ax = fig.add_subplot(1, num_img + 1, idx)
        ax.imshow(past_data[idx - 1])
    plt.suptitle(title, fontsize=30)
    plt.savefig(f"{title}")
    plt.close()


if __name__ == "__main__":
    train_set = MovingMNIST(
        root=".data/mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )

    # test dataset
    test_set = MovingMNIST(
        root=".data/mnist",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )

    args = easydict.EasyDict({
            "batch_size": 512,
            "input_size": 4096,
            "hidden_size": 2048,
            "latent_size": 1024,
            "learning_rate": 0.001,
            "num_layers": 2,
            "max_iter": 1000,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    })


    batch_size = args["batch_size"]
    input_size = args["input_size"]
    hidden_size = args["hidden_size"]
    latent_size = args["latent_size"]
    num_layers = args["num_layers"]

    # define LSTM-based VAE model
    model = LSTM_VAE(input_size, hidden_size, latent_size, num_layers)
    model.to(args.device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(test_set[0][0].shape)

    # train(args, model, train_loader, test_loader)
    #
    # torch.save(model.state_dict(), "model.pth")

    # load model
    # model_to_load = LSTM_VAE(input_size, hidden_size, latent_size, num_layers)
    # model_to_load.to(args.device)
    # model_to_load.load_state_dict(torch.load(f"model.pth"))
    # model_to_load.eval()
    #
    # # show results
    # ## past_data, future_data -> shape: (10,10)
    # future_data, past_data = train_set[0]
    #
    # ## reshape
    # example_size = past_data.size(0)
    # image_size = past_data.size(1), past_data.size(2)
    # past_data = past_data.view(example_size, -1).float().to(args.device)
    # _, recon_data, info = model_to_load(past_data.unsqueeze(0))
    #
    # nhw_orig = past_data.view(example_size, image_size[0], -1).cpu()
    # nhw_recon = (
    #     recon_data.squeeze(0)
    #     .view(example_size, image_size[0], -1)
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )
    #
    # imshow(nhw_orig, title=f"final_input")
    # imshow(nhw_recon, title=f"final_output")
    # plt.show()