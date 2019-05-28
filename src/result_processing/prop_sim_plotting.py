import os
import seaborn as sns
import matplotlib.pyplot as plt
import result_processing.compute_att as att
import pandas as pd


def make_reddit_prop_plt():
    sns.set()
    prop_expt = pd.DataFrame(att.process_propensity_experiment())

    prop_expt = prop_expt[['exog', 'plugin', 'one_step_tmle', 'very_naive']]
    prop_expt = prop_expt.rename(index=str, columns={'exog': 'Exogeneity',
                                         'very_naive': 'Unadjusted',
                                         'plugin': 'Plug-in',
                                         'one_step_tmle': 'TMLE'})
    prop_expt = prop_expt.set_index('Exogeneity')

    plt.figure(figsize=(4.75, 3.00))
    # plt.figure(figsize=(2.37, 1.5))
    sns.scatterplot(data=prop_expt, legend='brief', s=75)
    plt.xlabel("Exogeneity", fontfamily='monospace')
    plt.ylabel("NDE Estimate", fontfamily='monospace')
    plt.tight_layout()

    fig_dir = '../output/figures'
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir,'reddit_propensity.pdf'))


def main():
    make_reddit_prop_plt()


if __name__ == '__main__':
    main()