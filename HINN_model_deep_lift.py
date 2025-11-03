import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from keras import regularizers
from keras.optimizers import Adam
from keras.utils import plot_model

from captum.attr import IntegratedGradients
import plotly.express as px
import plotly.graph_objects as go

os.environ["KERAS_BACKEND"] = "torch"


def load_and_process_data():
    def preprocess(file_path, suffix):
        df = pd.read_csv(file_path)
        df.index = df.iloc[:, 0]
        df = df.drop(df.columns[0], axis=1)
        df.columns = [f"{col}_{suffix}" for col in df.columns]
        return df

    expression = preprocess("gene_data.csv", "expression")
    methy = preprocess("methyl_data.csv", "methy")
    snp = preprocess("snp_data.csv", "snp")

    demograph = pd.read_csv("demo_label_data.csv", usecols=range(7))
    demograph.index = demograph.iloc[:, 0]
    demograph = demograph.drop(demograph.columns[0], axis=1)
    demograph.columns = [f"{col}_demograph" for col in demograph.columns]

    label = pd.read_csv("demo_label_data.csv", usecols=[0, 8])
    label.index = label.iloc[:, 0]
    label = label.drop(label.columns[0], axis=1)
    label.columns = [f"{col}_label" for col in label.columns]

    data = pd.concat([snp, expression, methy, demograph, label], axis=1)

    data = data.dropna(subset=['MMSE_label'])

    return data


class PrimaryInputLayer(keras.layers.Layer):
    def __init__(self, units=50, output_dim=32, activation='sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.output_dim = output_dim
        self.activation_name = activation
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.mask = self.add_weight(
            shape=(self.units, self.output_dim),
            initializer="ones",
            trainable=False,
            name='mask'
        )
        self.w = self.add_weight(
            shape=(self.units, self.output_dim),
            initializer="glorot_normal",
            trainable=True,
            name='weights'
        )
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name='bias'
        )
        super().build(input_shape)

    def call(self, inputs):
        masked_weights = keras.ops.multiply(self.w, self.mask)
        return self.activation(keras.ops.matmul(inputs, masked_weights) + self.b)

    def set_mask(self, mask_array):
        if isinstance(mask_array, torch.Tensor):
            mask_array = mask_array.cpu().numpy()
        self.mask.assign(mask_array)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'output_dim': self.output_dim,
            'activation': self.activation_name
        })
        return config


class SecondaryInputLayer(keras.layers.Layer):
    def __init__(self, units=50, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        class IdentityInitializer(keras.initializers.Initializer):
            def __call__(self, shape, dtype=None):
                identity = torch.eye(shape[0], dtype=torch.float32)
                return identity

        self.mask = self.add_weight(
            shape=(self.units, self.units),
            initializer=IdentityInitializer(),
            trainable=False,
            name='mask'
        )
        self.w = self.add_weight(
            shape=(self.units, self.units),
            initializer="glorot_normal",
            trainable=True,
            name='weights'
        )
        super().build(input_shape)

    def call(self, inputs):
        masked_weights = keras.ops.multiply(self.w, self.mask)
        return keras.ops.matmul(inputs, masked_weights)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class MultiplicationInputLayer(keras.layers.Layer):
    def __init__(self, units=32, activation='sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name='bias'
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.activation(inputs + self.b)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation_name
        })
        return config


def preprocess_features(X_train, X_test, feature_types):
    X_train_processed = {}
    X_test_processed = {}

    for feat_type in feature_types:
        train_cols = [col for col in X_train.columns if f'_{feat_type}' in col]
        X_train_feat = X_train[train_cols].values
        X_test_feat = X_test[train_cols].values

        print(f"\n{feat_type.upper()} features:")
        print(f"  Train shape: {X_train_feat.shape}")

        imputer = SimpleImputer(strategy='mean')
        X_train_feat = imputer.fit_transform(X_train_feat)
        X_test_feat = imputer.transform(X_test_feat)

        if feat_type != 'snp':
            scaler = StandardScaler()
            X_train_feat = scaler.fit_transform(X_train_feat)
            X_test_feat = scaler.transform(X_test_feat)

        X_train_feat = np.nan_to_num(X_train_feat, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=0.0, neginf=0.0)

        X_train_processed[feat_type] = X_train_feat.astype(np.float32)
        X_test_processed[feat_type] = X_test_feat.astype(np.float32)

    return X_train_processed, X_test_processed


def train_model(X_train_list, y_train, X_val_list, y_val, model):
    history = model.fit(
        x=X_train_list,
        y=y_train,
        batch_size=64,
        epochs=3000,
        shuffle=True,
        validation_data=(X_val_list, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=50,
                mode="min",
                restore_best_weights=True,
                verbose=1
            )
        ],
        verbose=1
    )
    return history


def evaluate_model(model, X_test_list, y_test):
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)

    results = model.evaluate(x=X_test_list, y=y_test, verbose=2)

    predictions = model.predict(X_test_list, verbose=0)

    y_test_flat = y_test.flatten()
    pred_flat = predictions.flatten()

    if np.isnan(pred_flat).any():
        print("WARNING: NaN values found in predictions!")
        pred_flat = np.nan_to_num(pred_flat, nan=0.0)

    mse = np.mean((y_test_flat - pred_flat) ** 2)
    mae = np.mean(np.abs(y_test_flat - pred_flat))

    return {
        "results": results,
        "mse": mse,
        "mae": mae,
        "predictions": predictions
    }


class PytorchModelWrapper(nn.Module):
    def __init__(self, keras_model):
        super().__init__()
        self.keras_model = keras_model

    def forward(self, snp, methy, expr, demo):
        inputs = [
            snp.detach().cpu().numpy(),
            methy.detach().cpu().numpy(),
            expr.detach().cpu().numpy(),
            demo.detach().cpu().numpy()
        ]

        with torch.no_grad():
            pred = self.keras_model.predict(inputs, verbose=0)

        output = torch.from_numpy(pred).float()
        return self._attach_gradients(output, snp, methy, expr, demo)

    def _attach_gradients(self, output, snp, methy, expr, demo):
        dummy = (snp.mean() + methy.mean() + expr.mean() + demo.mean()) * 0.0
        return output + dummy


def interpret_model_simple(model, test_inputs, feature_names):
    test_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in test_inputs]

    importances = []

    for i, (tensor, names) in enumerate(zip(test_tensors, feature_names)):
        print(f"Processing input {i+1}/{len(test_tensors)}: {len(names)} features")

        baseline_pred = model.predict(test_inputs, verbose=0)

        feature_importance = np.zeros((tensor.shape[0], tensor.shape[1]))

        for j in range(min(tensor.shape[1], 100)):
            perturbed_inputs = [inp.copy() for inp in test_inputs]
            perturbed_inputs[i][:, j] += 0.1

            perturbed_pred = model.predict(perturbed_inputs, verbose=0)

            feature_importance[:, j] = np.abs(perturbed_pred.flatten() - baseline_pred.flatten())

        importances.append(torch.tensor(feature_importance, dtype=torch.float32))

    return tuple(importances)


def interpret_model_captum(model, test_inputs, baselines):
    try:
        wrapper = PytorchModelWrapper(model)
        wrapper.eval()

        test_tensors = tuple(
            torch.tensor(arr, dtype=torch.float32, requires_grad=True)
            for arr in test_inputs
        )

        baseline_tensors = tuple(
            torch.tensor(b, dtype=torch.float32)
            for b in baselines
        )

        ig = IntegratedGradients(wrapper)

        attributions = ig.attribute(
            test_tensors,
            baselines=baseline_tensors,
            n_steps=20,
            internal_batch_size=10
        )

        return attributions

    except Exception as e:
        print(f"Captum interpretation failed: {e}")
        return None


def interpret_model(model, test_inputs, baselines, feature_names):
    attributions = interpret_model_captum(model, test_inputs, baselines)

    if attributions is None:
        attributions = interpret_model_simple(model, test_inputs, feature_names)

    return attributions


def export_attributions(attributions, feature_names, save_path_prefix):
    if attributions is None:
        print("Skipping attribution export due to interpretation errors")
        return

    for i, name in enumerate(['snp', 'methy', 'gene', 'demo']):
        attr_data = attributions[i]
        if isinstance(attr_data, torch.Tensor):
            attr_data = attr_data.detach().cpu().numpy()

        df = pd.DataFrame(attr_data, columns=feature_names[i])
        df.to_csv(f"{save_path_prefix}_{name}.csv", index=False)
        print(f"Saved attributions for {name} to {save_path_prefix}_{name}.csv")


def filter_matrices_by_top_features(snp_list, methy_list, gene_list,
                                     sparse_methy, sparse_gene, sparse_pathway):
    snp_list = [s for s in snp_list if s in sparse_methy.index]
    methy_list = [m for m in methy_list if m in sparse_methy.columns and m in sparse_gene.index]
    gene_list = [g for g in gene_list if g in sparse_gene.columns and g in sparse_pathway.index]

    if not snp_list or not methy_list or not gene_list:
        print("Warning: No overlapping features found for Sankey diagram")
        return None, None, None

    subset_methy_matrix = sparse_methy.loc[snp_list, methy_list]
    subset_gene_matrix = sparse_gene.loc[methy_list, gene_list]
    subset_pathway_matrix = sparse_pathway.loc[gene_list, :]

    subset_methy_matrix = subset_methy_matrix.loc[
        subset_methy_matrix.any(axis=1),
        subset_methy_matrix.any(axis=0)
    ]
    subset_gene_matrix = subset_gene_matrix.loc[
        subset_gene_matrix.any(axis=1),
        subset_gene_matrix.any(axis=0)
    ]
    subset_pathway_matrix = subset_pathway_matrix.loc[
        subset_pathway_matrix.index.isin(subset_gene_matrix.columns)
    ]
    subset_pathway_matrix = subset_pathway_matrix.loc[
        subset_pathway_matrix.any(axis=1),
        subset_pathway_matrix.any(axis=0)
    ]

    return subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix


def summarize_connections(*matrices):
    connection_counts = [int(matrix.sum().sum()) for matrix in matrices if matrix is not None]
    labels = ["SNP-Methylation", "Methylation-Gene", "Gene-Pathway"]
    for label, count in zip(labels, connection_counts):
        print(f"Total connections ({label}): {count}")


def plot_sankey(subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix):
    if any(m is None for m in [subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix]):
        print("Skipping Sankey plot due to insufficient data")
        return

    snps = subset_methy_matrix.index.tolist()
    methys = subset_methy_matrix.columns.tolist()
    genes = subset_gene_matrix.columns.tolist()
    pathways = subset_pathway_matrix.columns.tolist()

    all_nodes = snps + methys + genes + pathways
    node_labels = [str(n) for n in all_nodes]
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    def create_links(matrix, source_list, target_list):
        sources, targets, values = [], [], []
        for src in source_list:
            for tgt in target_list:
                if matrix.loc[src, tgt] == 1:
                    sources.append(node_indices[src])
                    targets.append(node_indices[tgt])
                    values.append(1)
        return sources, targets, values

    s1, t1, v1 = create_links(subset_methy_matrix, snps, methys)
    s2, t2, v2 = create_links(subset_gene_matrix, methys, genes)
    s3, t3, v3 = create_links(subset_pathway_matrix, genes, pathways)

    sources = s1 + s2 + s3
    targets = t1 + t2 + t3
    values = v1 + v2 + v3

    fig = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color="blue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )
    ])

    fig.update_layout(title_text="The Sankey Diagram", font_size=10)
    fig.show()


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    data = load_and_process_data()

    X = data.drop(columns=[col for col in data.columns if 'MMSE_label' in col])
    y = data['MMSE_label']

    print(y.describe())

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_train_processed, X_test_processed = preprocess_features(
        X_train_df, X_test_df,
        ['snp', 'methy', 'expression', 'demograph']
    )

    X_train_snp = X_train_processed['snp']
    X_train_methy = X_train_processed['methy']
    X_train_exp = X_train_processed['expression']
    X_train_demo = X_train_processed['demograph']

    X_test_snp = X_test_processed['snp']
    X_test_methy = X_test_processed['methy']
    X_test_exp = X_test_processed['expression']
    X_test_demo = X_test_processed['demograph']

    X_train_list = [X_train_snp, X_train_methy, X_train_exp, X_train_demo]
    X_test_list = [X_test_snp, X_test_methy, X_test_exp, X_test_demo]

    y_train_array = y_train.values.reshape(-1, 1).astype(np.float32)
    y_test_array = y_test.values.reshape(-1, 1).astype(np.float32)

    sparse_methy = pd.read_csv("snp_methyl_matrix.csv", index_col=0)
    sparse_gene = pd.read_csv("methyl_gene_matrix.csv.zip", compression='zip', index_col=0)
    sparse_pathway = pd.read_csv("gene_pathway_matrix.csv", index_col=0)

    sparse_methy_array = np.nan_to_num(sparse_methy.values, nan=0.0).astype(np.float32)
    sparse_gene_array = np.nan_to_num(sparse_gene.values, nan=0.0).astype(np.float32)
    sparse_pathway_array = np.nan_to_num(sparse_pathway.values, nan=0.0).astype(np.float32)

    input_first_layer = Input(shape=(X_train_snp.shape[1],), name='snp_input')
    input_second_layer = Input(shape=(X_train_methy.shape[1],), name='methy_input')
    input_third_layer = Input(shape=(X_train_exp.shape[1],), name='expression_input')
    input_fourth_layer = Input(shape=(X_train_demo.shape[1],), name='demo_input')

    activation_function = 'relu'
    fully_activation_function = 'relu'
    kernel_initializer = 'he_normal'
    l2_reg = 0.001
    dense_nodes_1 = 64
    dense_nodes = 32
    drop_rate = 0.7

    # Layer 1: SNP -> Methylation
    primary_layer_1 = PrimaryInputLayer(
        units=X_train_snp.shape[1],
        output_dim=X_train_methy.shape[1],
        activation=activation_function,
        name='primary_1'
    )
    primary_output = primary_layer_1(input_first_layer)
    primary_layer_1.set_mask(sparse_methy_array)

    secondary_output = SecondaryInputLayer(
        units=X_train_methy.shape[1],
        name='secondary_1'
    )(input_second_layer)

    multiplication_result_1 = keras.ops.multiply(primary_output, secondary_output)
    multiplication_output = MultiplicationInputLayer(
        units=X_train_methy.shape[1],
        activation=activation_function,
        name='mult_1'
    )(multiplication_result_1)

    con_cat_layer_first = Dense(
        units=20,
        bias_initializer='zeros',
        activation=activation_function,
        name='dense_skip_1'
    )(input_first_layer)
    output_2 = Concatenate(name='concat_1')([multiplication_output, con_cat_layer_first])

    # Layer 2: Methylation -> Gene Expression
    second_layer = PrimaryInputLayer(
        units=X_train_methy.shape[1],
        output_dim=X_train_exp.shape[1],
        activation=activation_function,
        name='primary_2'
    )
    second_output = second_layer(multiplication_output)
    second_layer.set_mask(sparse_gene_array)

    third_output = SecondaryInputLayer(
        units=X_train_exp.shape[1],
        name='secondary_2'
    )(input_third_layer)

    #changing division to addition does not change anything (this is actually weird)
    
    #division_result_1 = keras.ops.add(third_output, second_output)
    #division_output = MultiplicationInputLayer(
    #    units=X_train_exp.shape[1], 
    #    activation=activation_function,
    #    name='mult_2'
    #)(division_result_1)
    
    epsilon = 1e-7

    division_result_1 = keras.ops.divide(third_output, second_output + epsilon)
    division_output = MultiplicationInputLayer(
        units=X_train_exp.shape[1],
        activation=activation_function,
        name='mult_2'
    )(division_result_1)

    con_cat_layer_sec = Dense(
        units=20,
        bias_initializer='zeros',
        activation=activation_function,
        name='dense_skip_2'
    )(output_2)
    output_3 = Concatenate(name='concat_2')([division_output, con_cat_layer_sec])

    # Layer 3: Gene Expression -> Pathway
    fourth_layer = PrimaryInputLayer(
        units=X_train_exp.shape[1],
        output_dim=sparse_pathway.shape[1],
        activation=activation_function,
        name='primary_3'
    )
    fourth_output = fourth_layer(division_output)
    fourth_layer.set_mask(sparse_pathway_array)

    con_cat_layer_third = Dense(
        units=20,
        bias_initializer='zeros',
        activation=activation_function,
        name='dense_skip_3'
    )(output_3)
    output_4 = Concatenate(name='concat_3')([fourth_output, con_cat_layer_third])

    x = output_4
    for i in range(3):
        x = BatchNormalization(name=f'bn_{i}')(x)
        x = Dense(
            units=dense_nodes_1,
            activation=fully_activation_function,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'dense_{i}'
        )(x)
        x = Dropout(drop_rate, name=f'dropout_{i}')(x)

    x = BatchNormalization(name='bn_final')(x)
    x = Dense(
        units=dense_nodes,
        activation=fully_activation_function,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_final'
    )(x)
    x = Dropout(drop_rate, name='dropout_final')(x)

    dense_fourth = Dense(
        units=20,
        activation=activation_function,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_pre_demo'
    )(x)
    demo_complete_layer = Concatenate(name='concat_demo')([dense_fourth, input_fourth_layer])

    x = BatchNormalization(name='bn_demo')(demo_complete_layer)
    x = Dense(
        units=dense_nodes_1,
        activation=fully_activation_function,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_demo'
    )(x)
    final_complete_layer = Dropout(drop_rate, name='dropout_demo')(x)

    outputs = Dense(
        units=1,
        activation='linear',
        kernel_initializer=kernel_initializer,
        name='output'
    )(final_complete_layer)

    model = Model(
        inputs=[input_first_layer, input_second_layer, input_third_layer, input_fourth_layer],
        outputs=outputs,
        name="HINN"
    )

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.0001, clipnorm=1.0),
        metrics=['mae']
    )

    print(model.summary())

    history = train_model(X_train_list, y_train_array, X_test_list, y_test_array, model)

    results = evaluate_model(model, X_test_list, y_test_array)
    print(f"\nTest Results:")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"  RMSE: {np.sqrt(results['mse']):.4f}")

    feature_names = [
        X_train_df.filter(like=s).columns.tolist()
        for s in ['_snp', '_methy', '_expression', '_demograph']
    ]

    baselines = [arr.mean(axis=0, keepdims=True) for arr in X_test_list]
    attributions = interpret_model(model, X_test_list, baselines, feature_names)

    if attributions is not None:
        export_attributions(attributions, feature_names, "MMSE")

        snp_list = [name.replace('_snp', '') for name in feature_names[0][:20]]
        methy_list = [name.replace('_methy', '') for name in feature_names[1][:100]]
        gene_list = [name.replace('_expression', '') for name in feature_names[2][:50]]

        subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix = filter_matrices_by_top_features(
            snp_list, methy_list, gene_list, sparse_methy, sparse_gene, sparse_pathway
        )

        if all(m is not None for m in [subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix]):
            summarize_connections(subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix)
            plot_sankey(subset_methy_matrix, subset_gene_matrix, subset_pathway_matrix)


if __name__ == "__main__":
    main()
