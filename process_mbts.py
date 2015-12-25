import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


def get_mbt_corrs(mbts):
    """
    :param list mbts: a list of individual MBT scores
    :returns pandas.DataFrame: pearon-correlation of individual score co-occurrences.
    """
    indiv_mbts = ['I', 'E', 'S', 'N', 'F', 'T', 'J', 'P']
    mbt_dict = dict()
    for val in indiv_mbts:
        mbt_dict[val] = [True if x.find(val) > -1 else False for x in mbts]
    return pd.DataFrame(mbt_dict).corr().ix[indiv_mbts, indiv_mbts]


def main():
    """
    Generate a bunch of graphs, and print out counts.
    """

    # Hard-coded type frequencies for the US from the Myers-Briggs site
    us_types = {'mbt': [('ISFJ', 13.8), ('ESFJ', 12.3),
                        ('ISTJ', 11.6), ('ISFP', 8.8),
                        ('ESTJ', 8.7), ('ESFP', 8.5),
                        ('ENFP', 8.1), ('ISTP', 5.4),
                        ('INFP', 4.4), ('ESTP', 4.3),
                        ('INTP', 3.3), ('ENTP', 3.2),
                        ('ENFJ', 2.5), ('INTJ', 2.1),
                        ('ENTJ', 1.8), ('INFJ', 1.5),
                        ('?', 0.0)],
                'indiv': [('I', 50.7), ('E', 49.3),
                          ('S', 73.3), ('N', 26.7),
                          ('F', 59.8), ('T', 40.2),
                          ('J', 54.1), ('P', 45.9)]}

    # Import and split data.
    df = pd.read_csv('data/MeFites & Myers-Briggs Types.csv')
    df.columns = ['timestamp', 'MBT']

    # set % based on total submitters
    # NOT the total number of submitted types.
    submission_count= df.shape[0]

    df['MBT'] = df['MBT'].apply(
        lambda x: x.replace('“It\'s always something different"', '?'))
    df['MBT'] = df['MBT'].apply(lambda x: x.replace(' ', ''))
    df['MBT'] = df['MBT'].apply(
        lambda x: ';'.join([y[0] + y[2:] if len(y) > 4 else y for y in x.split(';')]))

    # all_vals: _all_ of the individual types (multi-entry submissions split)
    all_vals = pd.Series(np.concatenate(df['MBT'].apply(lambda x: x.split(';'))))

    # Get bar plots and table text for US v. MeFi specific types

    a = pd.DataFrame(all_vals.value_counts()).reset_index()
    a.columns = ['MBT', 'Count']
    a['% MeFi'] = 100. * a['Count'] / submission_count  # a['Count'].sum()

    a = a.merge(pd.DataFrame.from_records(
        us_types['mbt'], columns=['MBT', '% US']), on=['MBT'])

    b = pd.melt(a, id_vars=['MBT'], value_vars=['% US', '% MeFi'])
    b.columns = ['MBT', 'Domain', 'Percentage']
    b['Domain'] = b['Domain'].apply(lambda x: x[2:])

    sns.set_context('talk', font_scale=0.75)
    f = sns.factorplot(x='MBT', y='Percentage', hue='Domain', data=b.sort_values(
        ['Domain', 'Percentage', 'MBT'], ascending=False), kind='bar', size=8)
    f.savefig('graphs/mefi_us_mbts.png', transparency=False)

    with open('tables/mefi_us_mbts.txt', 'w') as outfile:
        outfile.write(a.ix[:, ['MBT', '% US', '% MeFi', 'Count']].to_string(index=False))

    # Get bar plots and table text for US v. MeFi general types

    # all_vals_two: a list of _all_ individual letter categories.
    all_vals_two = pd.Series(np.concatenate(
        df['MBT'].apply(lambda x: list(x.replace(';', '').replace('?', '')))))

    c = pd.DataFrame(all_vals_two.value_counts()).reset_index()
    c.columns = ['MBT', 'Count']
    c['% MeFi'] = 100. * c['Count'] / (all_vals_two.shape[0] / 4)

    c = c.merge(pd.DataFrame.from_records(
        us_types['indiv'], columns=['MBT', '% US']), on=['MBT'])

    d = pd.melt(c, id_vars=['MBT'], value_vars=['% US', '% MeFi'])
    d.columns = ['MBT', 'Domain', 'Percentage']
    d['Domain'] = d['Domain'].apply(lambda x: x[2:])

    sns.set_context('talk', font_scale=1.)

    f, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
    for ax, order in zip(axes.flat, [('I', 'E'), ('S', 'N'), ('F', 'T'), ('J', 'P')]):
        g = sns.factorplot(x='MBT', y='Percentage', hue='Domain',
                           data=d, kind='bar',  order=order, ax=ax, legend=False)
        g.despine(left=True)
        ax.set(ylim=(0, 90.))
        ax.set_xlabel('')
    for i in [0, 2, 3]:
        f.axes[i].legend([])
    for i in [1, 3]:
        f.axes[i].set_ylabel('')
    for i in [0, 2]:
        f.axes[i].set_ylabel('Percentage')
    f.savefig('graphs/mefi_us_mbts_indiv.png', transparent=False)

    with open('tables/mefi_us_mbts_indiv.txt', 'w') as outfile:
        outfile.write(c.ix[[1, 6, 7, 0, 4, 3, 2, 5], ['MBT', '% US', '% MeFi', 'Count']].to_string(index=False))

    # Get correlation plot for general MeFi categories

    # all_vals_three: all specific values that aren't "?"
    all_vals_three = all_vals.ix[all_vals != '?']

    corrs = get_mbt_corrs(all_vals_three)
    fig, ax = plt.subplots()
    ax = sns.heatmap(corrs, vmax=0.4, vmin=-0.4, annot=True)
    fig.savefig('graphs/mefi_mbt_indiv_corrs.png', transparent=False)

    # Get correlation plot for general US categories.
    # This depends on values hard-coded into this file, so should be considered static.

    all_vals_four = np.repeat([x[0] for x in us_types['mbt']], [x[1]*10 for x in us_types['mbt']])
    corrs_two = get_mbt_corrs(all_vals_four)
    fig, ax = plt.subplots()
    ax = sns.heatmap(corrs_two, vmax=0.4, vmin=-0.4, annot=True)
    fig.savefig('graphs/us_mbt_indiv_corrs.png', transparent=False)


if __name__ == "__main__":
    main()
