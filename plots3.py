import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import hepherolib.data as data
import hepherolib.analysis as ana
import h5py

# Inicializa a análise
ana.start()
print("checked 1")

# Caminho para os datasets
basedir = '/home/miguel/Mestrado/HEPHeroLib/datasets_Run3/datasets'
period = '17'

# Carregar os datasets e exibir suas chaves
print("\nLoading datasets...")
try:
    datasets = data.read_files(basedir, period)
except ValueError as e:
    print(f"Erro ao carregar os datasets: {e}")
    exit(1)

print("\nDatasets carregados:")
for key in datasets:
    print(f"  - {key}")

# Definição das variáveis

vars = [ #'nElectron_RAW',#
         # 'nGenJet_RAW',#
         # 'nGenPart_RAW',#
         # 'GenMET_phi_RAW',#
         # 'GenMET_pt_RAW',#
         # 'nJet_RAW',#
         # 'nFatJet_RAW',#
         # 'nMuon_RAW', #
         # 'nTau_RAW', #
         # 'nSV_RAW', #
         # 'nIsoTrack_RAW',#
         # 'Pileup_nTrueInt',
         # 'MET_pt',
         # #'MET_phi',
         # #'Njets',
         # 'LeadingJet_pt',#
         #'SubLeadingJet_pt ',####################
          'ThirdLeadingJet_pt ',#
         # 'FourthLeadingJet_pt ' #
]
        
        
bin_vars = [#np.linspace(0,8,9),       #nElectron_RAW
            # np.linspace(2,20,19),     #nGenJet_RAW
            # np.linspace(20,105,86),   #nGenPart_RAW
            # np.linspace(-3.,3.,31),   #GenMET_phi_RAW
            # np.linspace(0,160,101),   #GenMET_pt_RAW
            # np.linspace(1,16,16),     #nJet_RAW
            # np.linspace(0,7,8),       #nFatJet_RAW
            # np.linspace(0,10,11),     #nMuon_RAW
            # np.linspace(0,6,7),       #nTau_RAW
            # np.linspace(0,10,11),     #nSV_RAW
            # np.linspace(0,12,13),     #nIsoTrack_RAW
            # np.linspace(0,70,71),     #Pileup_nTrueInt_RAW
            # np.linspace(0,1000,101),   #MET_pt
            #np.linspace(-3.,3.,31),   #MET_phi
            #np.linspace(0,17,18),     #Njets
            # np.linspace(50,1000,101),  #LeadingJet_pt
            #np.linspace(20,400,101),  #SubLeadingJet_pt
            np.linspace(20,200,101),  #ThirdLeadingJet_pt
            # np.linspace(20,200,101),  #FourthLeadingJet_pt
           ]


labels = [ #r"$nElectron_{RAW}$",
           # r"$nGenJet_{RAW}$",
           # r"$nGenPart_{RAW}$",
           # r"$GenMET_{\phi, RAW}$",
           # r"$GenMET_{pt, RAW}$",
           # r"$nJet$",
           # r"$nFatJet_{RAW}$",
           # r"$nMuon_{RAW}$",
           # r"$nTau_{RAW}$",
           # r"$nSV_{RAW}$",
           # r"$nIsoTrack_{RAW}$",
           # r"$Pileup_{nTrueInt}$"
           # r"$MET_{pt}$",
           #r"$MET_{\phi}$",
           #r"$Njets$",
           # r"$LeadingJet_{p_{T}}(GeV)$",
           #r"$SubLeadingJet_{p_{T}}(GeV)$",
            r"$ThirdLeadingJet_{p_{T}}(GeV)$",
           # r"$FourthLeadingJet_{p_{T}}(GeV)$",
         ]

print("checked 2")


def make_double_plots(dataset_1, dataset_2, dataset_3, dataset_4, vars, bin_vars, labels):
    """Função para gerar plots comparativos entre os datasets."""

    for idx, var in enumerate(vars):
        # print(f"\nProcessing variable: {var}")

        # Criação da figura e estrutura dos subplots
        fig = plt.figure(figsize=(8, 6))
        grid = [2, 1]
        gs1 = gs.GridSpec(grid[0], grid[1], height_ratios=[4, 1])

        # Subplot principal
        ax1 = plt.subplot(gs1[0])
        bins = bin_vars[idx]

        # Plotando os datasets
        try:
            ysgn1, errsgn1 = ana.step_plot(ax1, var, dataset_1, label='Run_3 H1000 a100', color='blue', weight="evtWeight", bins=bins, error=True, normalize=True)
            ysgn2, errsgn2 = ana.step_plot(ax1, var, dataset_2, label='Run_2 H1000 a100', color='red', weight="evtWeight", bins=bins, error=True, normalize=True)
            ysgn3, errsgn3 = ana.step_plot(ax1, var, dataset_3, label='Run_3 H400 a100', color='darkgreen', weight="evtWeight", bins=bins, error=True, normalize=True)
            ysgn4, errsgn4 = ana.step_plot(ax1, var, dataset_4, label='Run_2 H400 a100', color='lime', weight="evtWeight", bins=bins, error=True, normalize=True)

            # Configuração dos rótulos e estilo do gráfico principal
            ana.labels(ax1, ylabel="Events", xlabel=labels[idx])
            ana.style(ax1, lumi=0, energy_cm=0 , year=2022, legend_ncol=1, legend_fontsize=15, legend_loc='upper right')
            #ana.style(ax1, ylim=[0.02, 0.06], yticks=np.linspace(0.02, 0.06, 5), legend_fontsize=15) 

            # Subplot de comparação de barras de erro
            ax2 = plt.subplot(gs1[1], sharex=ax1)
            ana.ratio_plot(ax2, ysgn1, errsgn1, ysgn2, errsgn2, bins=bins, numerator="mc", color='blue')
            ana.labels(ax2, xlabel=labels[idx], ylabel=r'Run2/Run3')

            #ana.style(ax2, ylim=[0.8, 1.5], yticks=[0.8, 1.5], xgrid=True, ygrid=True) 
            #ana.style(ax2, yticks=[0., 1, 2], xgrid=True, ygrid=True)

            # Salvamento do plot
            os.makedirs('./plots/plots_Run3_Run2_com_erro_dissertacao2/', exist_ok=True)
            plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.95, hspace=0.05)
            plt.savefig(f'./plots/plots_Run3_Run2_com_erro_dissertacao2/{var}.png')
            plt.close(fig)

        except KeyError as e:
            print(f"Variável ausente ao tentar plotar {var}: {e}")
        except Exception as e:
            print(f"Erro ao processar {var}: {e}")

# Execução da função com os datasets definidos
print("\nIniciando a geração dos plots...")
try:
    make_double_plots(
        datasets.get('Signal_Run3_22_1000_100', pd.DataFrame()),
        datasets.get('Signal_10_6_10_1000_100_matheus', pd.DataFrame()),
        datasets.get('Signal_Run3_22_400_100', pd.DataFrame()),
        datasets.get('Signal_quarks_400_100_matheus', pd.DataFrame()),
        vars, bin_vars, labels
    )
except Exception as e:
    print(f"Erro na execução dos plots: {e}")

print("\nProcesso concluído.")
