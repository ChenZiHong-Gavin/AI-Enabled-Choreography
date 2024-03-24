import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

    def forward(self, x):
        """
        :param x: batch_size x seq_len x input_dim
        """
        _, (hidden, cell) = self.lstm(x)
        return (hidden, cell)


class Decoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_size * num_layers, output_size)

    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.linear(output)
        last_frame = prediction[:, -1, :]
        return last_frame, (hidden, cell)


class LSTM_VAE(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers):
        """
        :param input_size: batch_size x seq_len x input_dim
        :param hidden_size: output size of the LSTM layer
        :param latent_size: latent z-layer size
        :param num_layers: number of LSTM layers
        """
        super(LSTM_VAE, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # LSTM VAE
        self.lstm_encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers)
        self.lstm_decoder = Decoder(
            input_size=latent_size,
            output_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers)

        self.mu = torch.nn.Linear(hidden_size * num_layers, latent_size)
        self.logvar = torch.nn.Linear(hidden_size * num_layers, latent_size)
        self.linear = torch.nn.Linear(latent_size, hidden_size * num_layers)


    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        return mu + noise * std

    def forward(self, x, label):
        """
        :param x: batch_size x seq_len x input_dim
        """
        batch_size, seq_len, input_dim = x.shape

        # encode input space to hidden space
        enc_hidden = self.lstm_encoder(x) # num_layers x batch_size x hidden_size
        enc_hidden = enc_hidden[0].view(batch_size, self.hidden_size * self.num_layers).to(self.device) # batch_size x (hidden_size * num_layers)

        # hidden space to latent space
        mu = self.mu(enc_hidden)
        logvar = self.logvar(enc_hidden)
        z = self.reparametize(mu, logvar)

        # initialize hidden state as inputs
        h_ = self.linear(z)
        h_ = h_.contiguous().view(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # decode latent space to input space
        z = z.repeat(1, seq_len, 1) # batch_size x seq_len x latent_size
        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)

        # initialize hidden state
        hidden = (h_.contiguous(), h_.contiguous())

        reconstruct_output, _ = self.lstm_decoder(z, hidden)

        x_hat = reconstruct_output
        if label is None:
            return x_hat

        label = label.squeeze(1)

        # calculate loss
        losses = self.loss_function(x_hat, label, mu, logvar)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD"],
        )

        return m_loss, reconstruct_output, (recon_loss, kld_loss)


    def loss_function(self, recons, input, mu, log_var):
        """
        :param recons: reconstructed output
        :param input: original input
        :param mu: latent mean
        :param log_var: latent log variance
        Computes the VAE loss function
        """
        kld_weight = 1
        recons_loss = torch.nn.functional.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss

        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach()
        }




