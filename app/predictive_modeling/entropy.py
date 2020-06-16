import pandas as pd
from scipy import stats


def entropysq(df, control, target):
    classlist = df[control].unique().tolist()
    out = pd.DataFrame(columns=[control, 'Entropy'])
    out[str(control)] = classlist
    out.Entropy = list(map(
        lambda a: stats.entropy(df.loc[df[str(control)] == a, target].tolist()), classlist))
    return out
