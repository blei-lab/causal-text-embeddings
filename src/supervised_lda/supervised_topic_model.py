import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SupervisedTopicModel(nn.Module):
    def __init__(self, num_topics, vocab_size, num_documents, t_hidden_size=800, theta_act='relu', enc_drop=0., outcome_linear_map=True):
        super(SupervisedTopicModel, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.num_documents = num_documents
        self.t_hidden_size = t_hidden_size
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        self.theta_act = self.get_activation(theta_act)
        self.outcome_linear_map = outcome_linear_map
        
        ## define the matrix containing the topic embeddings
        self.alphas = nn.Parameter(torch.randn(vocab_size, num_topics))

        if self.outcome_linear_map:
            ## define linear regression weights for predicting expected outcomes for treated
            self.w_expected_outcome_treated = nn.Linear(num_topics, 1)

            ## define linear regression weights for predicting expected outcomes for untreated
            self.w_expected_outcome_untreated = nn.Linear(num_topics, 1)
        else:
            self.f_outcome_treated = nn.Sequential(
                nn.Linear(num_topics, t_hidden_size), 
                self.theta_act,
                # nn.BatchNorm1d(t_hidden_size),
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
                # nn.BatchNorm1d(t_hidden_size),
                nn.Linear(t_hidden_size,1)
            )
            self.f_outcome_untreated = nn.Sequential(
                nn.Linear(num_topics, t_hidden_size), 
                self.theta_act,
                # nn.BatchNorm1d(t_hidden_size),
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
                # nn.BatchNorm1d(t_hidden_size),
                nn.Linear(t_hidden_size,1)
            )
        ## define linear regression weights for predicting binary treatment label
        self.w_treatment = nn.Linear(num_topics,1)
        
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
                self.theta_act,
                nn.BatchNorm1d(t_hidden_size),
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
                nn.BatchNorm1d(t_hidden_size)
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        beta = F.softmax(self.alphas, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta) 
        preds = torch.log(res+1e-6)
        return preds

    def predict_treatment(self, theta):
        logits = self.w_treatment(theta)
        return logits

    def predict_outcome_st_treat(self, theta, treatment_labels):
        treated_indices = [treatment_labels == 1]
        theta_treated = theta[treated_indices]

        if not self.outcome_linear_map:
            expected_outcome_treated = self.f_outcome_treated(theta_treated)
        else:
            expected_outcome_treated = self.w_expected_outcome_treated(theta_treated) 

        return expected_outcome_treated

    def predict_outcome_st_no_treat(self, theta, treatment_labels):
        untreated_indices = [treatment_labels == 0]
        theta_untreated = theta[untreated_indices]

        if not self.outcome_linear_map:
            expected_outcome_untreated = self.f_outcome_untreated(theta_untreated)
        else:
            expected_outcome_untreated = self.w_expected_outcome_untreated(theta_untreated)
            
        return expected_outcome_untreated


    def forward(self, bows, normalized_bows, treatment_labels, outcomes, dtype='real', use_supervised_loss=True):
        ## get \theta
        theta, kld_theta = self.get_theta(normalized_bows)
        beta = self.get_beta()

        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        
        ## get reconstruction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        recon_loss = recon_loss.mean()

        supervised_loss=None
        if use_supervised_loss:

            #get treatment loss 
            treatment_logits = self.predict_treatment(theta).squeeze()
            treatment_loss = bce_loss(treatment_logits, treatment_labels)

            #get expected outcome loss
            treated  = [treatment_labels == 1]
            untreated = [treatment_labels == 0]
            outcomes_treated = outcomes[treated]
            outcomes_untreated = outcomes[untreated]
            expected_treated = self.predict_outcome_st_treat(theta, treatment_labels).squeeze()
            expected_untreated = self.predict_outcome_st_no_treat(theta, treatment_labels).squeeze()

            if dtype == 'real':
                outcome_loss_treated = mse_loss(expected_treated,outcomes_treated)
                outcome_loss_untreated = mse_loss(expected_treated,outcomes_treated)
            else:
                outcome_loss_treated = bce_loss(expected_treated,outcomes_treated)
                outcome_loss_untreated = bce_loss(expected_treated,outcomes_treated)

            supervised_loss = treatment_loss + outcome_loss_treated + outcome_loss_untreated

        return recon_loss, supervised_loss, kld_theta

