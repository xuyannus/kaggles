from subprocess import call

from sklearn.tree import export_graphviz


def plot_first_tree(rf_model, feature_names, file_name_id):
    export_graphviz(rf_model.estimators_[0],
                    out_file='tree.dot',
                    feature_names=feature_names,
                    precision=2,
                    filled=True,
                    rounded=True)
    call(['/usr/local/bin/dot', '-Tpng', 'tree.dot', '-o',
          './house_price/tree_{}.png'.format(file_name_id), '-Gdpi=600'])
