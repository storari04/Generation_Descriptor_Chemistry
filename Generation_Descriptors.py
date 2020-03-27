# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors

dataset_type = 0 # 0:smiles 1:sdf 2:mordred_smiles

dataset = pd.read_csv('molecules_with_boiling_point.csv', index_col=0)  # SMILES 付きデータセットの読み込み
smiles = dataset.iloc[:, 0]  # 分子の SMILES
y = dataset.iloc[:, 1]  # 物性・活性などの目的変数

if dataset_type == 0:
    # 計算する記述子名の取得
    descriptor_names = []
    for descriptor_information in Descriptors.descList:
        descriptor_names.append(descriptor_information[0])
    print('計算する記述子の数 :', len(descriptor_names))

    # 記述子の計算
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = []  # ここに計算された記述子の値を追加
    print('分子の数 :', len(smiles))
    for index, smiles_i in enumerate(smiles):
        print(index + 1, '/', len(smiles))
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors = pd.DataFrame(descriptors, index=dataset.index, columns=descriptor_names)

if dataset_type == 1:
    sdf = Chem.SDMolSupplier('boiling_point.sdf')  # sdf ファイルの読み込み

    # 計算する記述子名の取得
    descriptor_names = []
    for descriptor_information in Descriptors.descList:
        descriptor_names.append(descriptor_information[0])
    print('計算する記述子の数 :', len(descriptor_names))

    # 記述子の計算
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    # 分子ごとに、リスト型の変数 y に物性値を、descriptors に計算された記述子の値を、smiles に SMILES を追加
    descriptors, y, smiles = [], [], []
    print('分子の数 :', len(sdf))
    for index, molecule in enumerate(sdf):
        print(index + 1, '/', len(sdf))
        y.append(float(molecule.GetProp(y_name)))
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
        smiles.append(Chem.MolToSmiles(molecule))
    descriptors = pd.DataFrame(descriptors, index=smiles, columns=descriptor_names)
    y = pd.DataFrame(y, index=smiles, columns=[y_name])

if dataset_type == 2:
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    calc = Calculator(descriptors, ignore_3D=True)
    descriptors = calc.pandas(mols)

    descriptors = descriptors.astype(str)
    masks = descriptors.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
    descriptors = descriptors[~masks]
    descriptors = descriptors.astype(float)

    y = pd.DataFrame(y, index=smiles, columns=[y_name])

if dataset_type == 3:
    from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
    from descriptastorus.descriptors import rdDescriptors
    from descriptastorus.descriptors import rdNormalizedDescriptors
    gen1 = MakeGenerator(('rdkit2d', 'Morgan3counts'))
    gen2 = rdDescriptors.RDKit2D()
    gen3 = rdNormalizedDescriptors.RDKit2DNormalized()

    data1 = gen1.process(smiles)
    data2 = gen2.process(smiles)
    data3 = gen3.process(smiles)
    for col in gen1.GetColumns():
        y_name.append(col)
    y = pd.DataFrame(y, index=smiles, columns=[y_name])

if dataset_type == 4: #3D Descriptors
    from e3fp.fingerprint.generate import fp, fprints_dict_from_mol
    from e3fp.conformer.generate import generate_conformers
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    optimize_mols = []
    for mol in mols:
        mh = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mh)
        AllChem.MMFFOptimizeMolecule(mh)
        optimize_mols.append(mh)
    fpdicts = [fprints_dict_from_mol(mol) for mol in optimize_mols]
    # if molecule has multiple conformers the function will generate multiple fingerprints.
    fps = [fp[5][0] for fp in fpdicts]
    # convert to rdkit fp from e3fp fingerprint
    binfp = [fp.fold().to_rdkit() for fp in fps]
    y = pd.DataFrame(binfp, index=smiles)
# 保存
descriptors_with_y = pd.concat([y, descriptors], axis=1)  # y と記述子を結合
descriptors_with_y.to_csv('descriptors_with_y.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
