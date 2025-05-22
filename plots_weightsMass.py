import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import hepherolib.data as data
import hepherolib.analysis as ana
import h5py

ana.start() # Starts the HepHero analysis environment


basedir = '/home/miguel/Mestrado/HEPHeroLib/datasets_Run3/datasets' # Define base directory where dataset files are located
period = '00_22' # Data period, for NanoAOD tested, it's 0_22 (2022)

#===================================================================
#===================================================================
# You need to choose the sets and the var of the histograms below
#===================================================================
#===================================================================


sets="Signal_400_100";      bin_Hmass=np.linspace(310, 490, 21);    bin_pT=np.linspace(0, 400, 21);     bin_eta=np.linspace(-3, 3, 21); bin_amass_xx=np.linspace(90, 120, 21); bin_ZMass_qqbar=np.linspace(70, 110, 21)
#sets="Signal_1000_100";     bin_Hmass=np.linspace(500, 1500, 21);   bin_pT=np.linspace(0, 1100, 21);    bin_eta=np.linspace(-3, 3, 21);  bin_amass_xx=np.linspace(90, 120, 21); bin_ZMass_qqbar=np.linspace(70, 110, 21)
 

#Which variable do you want to work with?
# 0 = all; 1 = HMass_qq_chichi; 2 = qq_pt; 3 = qq_eta
variable = 0


#==================================================================
#==================================================================
#You need to choose the sets and the limits of the histograms above
#==================================================================
#==================================================================


# Read datasets from files
datasets = data.read_files(basedir, period)
datasets_vec = data.read_files(basedir, period, mode="vectors")

def make_plot(dataset_name, var, bin_vars, custom_weight_groups=None, colors=None, labels=None, line_styles=None, normalize=False, xlabel=None, save_name=None):

    """
    Generates a plot with the option to combine irregular weights for each line.
    Args:
    dataset_name (str): Name of the dataset to use for plotting.
    var (str): The variable to be plotted.
    bin_vars (array): Bins for the variable to be plotted.
    custom_weight_groups (list): List of lists, where each sublist contains the names of weights to be combined for one line.
    colors (list, optional): List of colors for each line.
    labels (list, optional): List of labels for each line in the legend.
    line_styles (list, optional): List of line styles for each line.
    normalize (bool): If True, normalizes the histogram by the area.
    xlabel (str, optional): Label for the x-axis.
    save_name (str, optional): The name for saving the plot.
    """

    # Load the dataset based on the dataset name
    dataset = datasets[dataset_name]
    dataset_vec = datasets_vec[dataset_name]

    num_weights = 125 # Number of weight variables to use

    #Defining weight
    for i in range(num_weights):
        peso_nome = f'signal_weight{i}'  # Create a weight variable name, e.g., 'signal_weight0', 'signal_weight1', ...
        dataset[peso_nome] = dataset['evtWeight'] * dataset_vec['param_variation_weights'][:, i] # Multiply the event weight by the parameter variation weights

    # Initialize the plot
    fig = plt.figure(figsize=(7, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    # Prepare dynamic text for the legend using dataset parameters
    dataset_name = dataset_name.split('_') if save_name else ['Unknown', '0', '0']
    mH, ma, mchi = dataset_name[1], dataset_name[2], '45'
    dynamic_text = fr"$\mathrm{{b\bar{{b}}H→Za→q\bar{{q}}χ\bar{{\chi}}~(m_{{H,a,χ}}={mH},{ma},{mchi}~GeV)}}$"

    # Loop through the custom weight groups to plot each line
    for idx, weight_group in enumerate(custom_weight_groups): # Loop pelos grupos de pesos
        combined_data = np.zeros(len(bin_vars) - 1) # Combina os pesos especificados no grupo
        total_weight = 0  # Soma total dos pesos

        # Loop through each weight in the weight group and add it to the combined data
        for weight in weight_group:
            counts, _ = np.histogram(
                dataset[var], 
                bins=bin_vars, 
                weights=dataset[weight],
            )
            combined_data += counts
            total_weight += np.sum(dataset[weight])  # Soma total de pesos para normalização

        # Normalize by the total weight (or area)
        if normalize and total_weight > 0:
            combined_data /= total_weight  

        # Set visual properties for the line
        label = labels[idx] if labels else f'Line {idx+1}'
        color = colors[idx] if colors else 'blue'
        line_style = line_styles[idx] if line_styles else '-'
        
         # Plot the data using step lines
        plt.step(bin_vars, np.append([combined_data[0]], combined_data), 
                 label=label, color=color, linestyle=line_style, linewidth=1.5)

    ax1.legend(fontsize=18)

    #layout
    ana.labels(ax1, ylabel="Normalized to unit", xlabel=xlabel if xlabel else r"---")
    ana.style(ax1, energy_cm=13.6, year=2022, legend_ncol=1, xticklabels=True, ylog=True, ylim=[0, 200])
    ax1.legend(fontsize=18, loc=(0.6, 0.45))
    os.makedirs('./plots_weight_extras/combined/'+sets+'/'+var, exist_ok=True)
    plt.subplots_adjust(left=0.1, bottom=0.115, right=0.96, top=0.95)
    x_position = bin_vars[0] + 0.04 * (bin_vars[-1] - bin_vars[0])
    plt.text(x_position, 40, dynamic_text, fontsize=18, color='indianred')
    save_path = f'./plots_weight_extras/combined/'+sets+'/'+var+f'/{save_name if save_name else "combined_plot_normalized"}.pdf'
    plt.savefig(save_path) # Save the plot to file
    #plt.show()


if variable == 1 or variable == 0:
    # Chamando a função para tan(beta) Mass H
    make_plot(
        dataset_name=sets,  # Nome do dataset
        var="HMass_qq_chichi",  # Variável a ser plotada
        bin_vars=bin_Hmass,  # Intervalo de bins
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Diferentes cores para os intervalos
        labels=['tan(β) = 5', 'tan(β) = 10', 'tan(β) = 15', 'tan(β) = 20', 'tan(β) = 30'],  # Legendas para cada linha
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        xlabel=r"$M(q,\bar{{q}},\chi,\bar{{\chi}})~GeV$",  # Label para o eixo x
        save_name=f"{sets}_HMass_tan",  # Nome do arquivo de saída
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 24)],  # Intervalo de pesos para tab(β) = 5
            [f'signal_weight{i}' for i in range(25, 49)],  # Intervalo de pesos para tab(β) = 10
            [f'signal_weight{i}' for i in range(50, 74)],  # Intervalo de pesos para tab(β) = 15
            [f'signal_weight{i}' for i in range(75, 99)],  # Intervalo de pesos para tab(β) = 20
            [f'signal_weight{i}' for i in range(100, 124)],  # Intervalo de pesos para tab(β) = 30
        ]
    )
    # Chamando a função para tan(beta) Mass a
    make_plot(
        dataset_name=sets,  # Nome do dataset
        var="amass_xx",  # Variável a ser plotada
        bin_vars=bin_amass_xx,  # Intervalo de bins
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Diferentes cores para os intervalos
        labels=['tan(β) = 5', 'tan(β) = 10', 'tan(β) = 15', 'tan(β) = 20', 'tan(β) = 30'],  # Legendas para cada linha
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        xlabel=r"$M(\chi,\bar{{\chi}})~GeV$",  # Label para o eixo x
        save_name=f"{sets}_amass_tan",  # Nome do arquivo de saída
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 24)],  # Intervalo de pesos para tab(β) = 5
            [f'signal_weight{i}' for i in range(25, 49)],  # Intervalo de pesos para tab(β) = 10
            [f'signal_weight{i}' for i in range(50, 74)],  # Intervalo de pesos para tab(β) = 15
            [f'signal_weight{i}' for i in range(75, 99)],  # Intervalo de pesos para tab(β) = 20
            [f'signal_weight{i}' for i in range(100, 124)],  # Intervalo de pesos para tab(β) = 30
        ]
    )

    # Chamando a função para tan(beta) Mass z
    make_plot(
        dataset_name=sets,  # Nome do dataset
        var="ZMass_qqbar",  # Variável a ser plotada
        bin_vars=bin_ZMass_qqbar,  # Intervalo de bins
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Diferentes cores para os intervalos
        labels=['tan(β) = 5', 'tan(β) = 10', 'tan(β) = 15', 'tan(β) = 20', 'tan(β) = 30'],  # Legendas para cada linha
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        xlabel=r"$M(q,\bar{q})~GeV$",  # Label para o eixo x
        save_name=f"{sets}_ZMass_qqbar_tan",  # Nome do arquivo de saída
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 24)],  # Intervalo de pesos para tab(β) = 5
            [f'signal_weight{i}' for i in range(25, 49)],  # Intervalo de pesos para tab(β) = 10
            [f'signal_weight{i}' for i in range(50, 74)],  # Intervalo de pesos para tab(β) = 15
            [f'signal_weight{i}' for i in range(75, 99)],  # Intervalo de pesos para tab(β) = 20
            [f'signal_weight{i}' for i in range(100, 124)],  # Intervalo de pesos para tab(β) = 30
        ]
    )

    # Chamando a função para sin(Theta) Mass H
    make_plot(
        dataset_name=sets,
        var="HMass_qq_chichi",
        bin_vars=bin_Hmass,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['sin('r'$\theta$) = 0.1', 'sin('r'$\theta$) = 0.3', 'sin('r'$\theta$) = 0.5', 'sin('r'$\theta$) = 0.7', 'sin('r'$\theta$) = 0.9'],  # Rótulos para cada linha
        xlabel=r"$M(q,\overline{q},\chi,\overline{\chi})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_HMass_sin", #Nome do arquivo .pdf
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 4)] + [f'signal_weight{i}' for i in range(25, 29)] + [f'signal_weight{i}' for i in range(50, 54)]  + [f'signal_weight{i}' for i in range(75, 79)] + [f'signal_weight{i}' for i in range(100, 104)],  # Sin(theta) = 0.1
            [f'signal_weight{i}' for i in range(5, 9)] + [f'signal_weight{i}' for i in range(30, 34)] + [f'signal_weight{i}' for i in range(55, 59)]  + [f'signal_weight{i}' for i in range(80, 84)] + [f'signal_weight{i}' for i in range(105, 109)],  # Sin(theta) = 0.3
            [f'signal_weight{i}' for i in range(10, 14)] + [f'signal_weight{i}' for i in range(35, 39)] + [f'signal_weight{i}' for i in range(60, 64)]  + [f'signal_weight{i}' for i in range(85, 89)] + [f'signal_weight{i}' for i in range(110, 114)],  # Sin(theta) = 0.5
            [f'signal_weight{i}' for i in range(15, 19)] + [f'signal_weight{i}' for i in range(40, 44)] + [f'signal_weight{i}' for i in range(65, 69)]  + [f'signal_weight{i}' for i in range(90, 94)] + [f'signal_weight{i}' for i in range(115, 119)],  # Sin(theta) = 0.7
            [f'signal_weight{i}' for i in range(20, 24)] + [f'signal_weight{i}' for i in range(45, 49)] + [f'signal_weight{i}' for i in range(70, 74)]  + [f'signal_weight{i}' for i in range(95, 99)] + [f'signal_weight{i}' for i in range(120, 124)],  # Sin(theta) = 0.9
        ]

    )

    # Chamando a função para sin(Theta) Mass a
    make_plot(
        dataset_name=sets,
        var="amass_xx",
        bin_vars=bin_amass_xx,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['sin('r'$\theta$) = 0.1', 'sin('r'$\theta$) = 0.3', 'sin('r'$\theta$) = 0.5', 'sin('r'$\theta$) = 0.7', 'sin('r'$\theta$) = 0.9'],  # Rótulos para cada linha
        xlabel=r"$M(\chi,\overline{\chi})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_amass_sin", #Nome do arquivo .pdf
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 4)] + [f'signal_weight{i}' for i in range(25, 29)] + [f'signal_weight{i}' for i in range(50, 54)]  + [f'signal_weight{i}' for i in range(75, 79)] + [f'signal_weight{i}' for i in range(100, 104)],  # Sin(theta) = 0.1
            [f'signal_weight{i}' for i in range(5, 9)] + [f'signal_weight{i}' for i in range(30, 34)] + [f'signal_weight{i}' for i in range(55, 59)]  + [f'signal_weight{i}' for i in range(80, 84)] + [f'signal_weight{i}' for i in range(105, 109)],  # Sin(theta) = 0.3
            [f'signal_weight{i}' for i in range(10, 14)] + [f'signal_weight{i}' for i in range(35, 39)] + [f'signal_weight{i}' for i in range(60, 64)]  + [f'signal_weight{i}' for i in range(85, 89)] + [f'signal_weight{i}' for i in range(110, 114)],  # Sin(theta) = 0.5
            [f'signal_weight{i}' for i in range(15, 19)] + [f'signal_weight{i}' for i in range(40, 44)] + [f'signal_weight{i}' for i in range(65, 69)]  + [f'signal_weight{i}' for i in range(90, 94)] + [f'signal_weight{i}' for i in range(115, 119)],  # Sin(theta) = 0.7
            [f'signal_weight{i}' for i in range(20, 24)] + [f'signal_weight{i}' for i in range(45, 49)] + [f'signal_weight{i}' for i in range(70, 74)]  + [f'signal_weight{i}' for i in range(95, 99)] + [f'signal_weight{i}' for i in range(120, 124)],  # Sin(theta) = 0.9
        ]

    )


    # Chamando a função para sin(Theta) Mass z
    make_plot(
        dataset_name=sets,
        var="ZMass_qqbar",
        bin_vars=bin_ZMass_qqbar,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['sin('r'$\theta$) = 0.1', 'sin('r'$\theta$) = 0.3', 'sin('r'$\theta$) = 0.5', 'sin('r'$\theta$) = 0.7', 'sin('r'$\theta$) = 0.9'],  # Rótulos para cada linha
        xlabel=r"$M(q,\overline{q})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_ZMass_qqbar_sin", #Nome do arquivo .pdf
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 4)] + [f'signal_weight{i}' for i in range(25, 29)] + [f'signal_weight{i}' for i in range(50, 54)]  + [f'signal_weight{i}' for i in range(75, 79)] + [f'signal_weight{i}' for i in range(100, 104)],  # Sin(theta) = 0.1
            [f'signal_weight{i}' for i in range(5, 9)] + [f'signal_weight{i}' for i in range(30, 34)] + [f'signal_weight{i}' for i in range(55, 59)]  + [f'signal_weight{i}' for i in range(80, 84)] + [f'signal_weight{i}' for i in range(105, 109)],  # Sin(theta) = 0.3
            [f'signal_weight{i}' for i in range(10, 14)] + [f'signal_weight{i}' for i in range(35, 39)] + [f'signal_weight{i}' for i in range(60, 64)]  + [f'signal_weight{i}' for i in range(85, 89)] + [f'signal_weight{i}' for i in range(110, 114)],  # Sin(theta) = 0.5
            [f'signal_weight{i}' for i in range(15, 19)] + [f'signal_weight{i}' for i in range(40, 44)] + [f'signal_weight{i}' for i in range(65, 69)]  + [f'signal_weight{i}' for i in range(90, 94)] + [f'signal_weight{i}' for i in range(115, 119)],  # Sin(theta) = 0.7
            [f'signal_weight{i}' for i in range(20, 24)] + [f'signal_weight{i}' for i in range(45, 49)] + [f'signal_weight{i}' for i in range(70, 74)]  + [f'signal_weight{i}' for i in range(95, 99)] + [f'signal_weight{i}' for i in range(120, 124)],  # Sin(theta) = 0.9
        ]

    )



    # Chamando a função para lan3 Mass H
    make_plot(
        dataset_name=sets,
        var="HMass_qq_chichi", 
        bin_vars=bin_Hmass,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['$\lambda_{3} = 0.3$', '$\lambda_{3} = 0.4$', '$\lambda_{3} = 0.5$', '$\lambda_{3} = 0.6$', '$\lambda_{3} = 0.7$'],  # Rótulos para cada linha
        xlabel=r"$M(q,\overline{q},\chi,\overline{\chi})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_HMass_lambda3",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(125) if i % 5 == 0],  # Múltiplos de 5
            [f'signal_weight{i}' for i in range(125) if i % 5 == 1],  # Múltiplos de 5+1
            [f'signal_weight{i}' for i in range(125) if i % 5 == 2],  # Múltiplos de 5+2
            [f'signal_weight{i}' for i in range(125) if i % 5 == 3],  # Múltiplos de 5+3
            [f'signal_weight{i}' for i in range(125) if i % 5 == 4],  # Múltiplos de 5+4
        ]
    )

    # Chamando a função para lan3 Mass a
    make_plot(
        dataset_name=sets,
        var="amass_xx", 
        bin_vars=bin_amass_xx,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['$\lambda_{3} = 0.3$', '$\lambda_{3} = 0.4$', '$\lambda_{3} = 0.5$', '$\lambda_{3} = 0.6$', '$\lambda_{3} = 0.7$'],  # Rótulos para cada linha
        xlabel=r"$M(\chi,\overline{\chi})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_amass_lambda3",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(125) if i % 5 == 0],  # Múltiplos de 5
            [f'signal_weight{i}' for i in range(125) if i % 5 == 1],  # Múltiplos de 5+1
            [f'signal_weight{i}' for i in range(125) if i % 5 == 2],  # Múltiplos de 5+2
            [f'signal_weight{i}' for i in range(125) if i % 5 == 3],  # Múltiplos de 5+3
            [f'signal_weight{i}' for i in range(125) if i % 5 == 4],  # Múltiplos de 5+4
        ]
    )


    # Chamando a função para lan3 Mass z
    make_plot(
        dataset_name=sets,
        var="ZMass_qqbar", 
        bin_vars=bin_ZMass_qqbar,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['$\lambda_{3} = 0.3$', '$\lambda_{3} = 0.4$', '$\lambda_{3} = 0.5$', '$\lambda_{3} = 0.6$', '$\lambda_{3} = 0.7$'],  # Rótulos para cada linha
        xlabel=r"$M(q,\overline{q})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_ZMass_qqbar_lambda3",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(125) if i % 5 == 0],  # Múltiplos de 5
            [f'signal_weight{i}' for i in range(125) if i % 5 == 1],  # Múltiplos de 5+1
            [f'signal_weight{i}' for i in range(125) if i % 5 == 2],  # Múltiplos de 5+2
            [f'signal_weight{i}' for i in range(125) if i % 5 == 3],  # Múltiplos de 5+3
            [f'signal_weight{i}' for i in range(125) if i % 5 == 4],  # Múltiplos de 5+4
        ]
    )


if variable == 2 or variable == 0:
    # Chamando a função para tan(beta) pT qq
    make_plot(
        dataset_name=sets,  # Nome do dataset
        var="qq_pt",  # Variável a ser plotada
        bin_vars=bin_pT,  # Intervalo de bins
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Diferentes cores para os intervalos
        labels=['tan(β) = 5', 'tan(β) = 10', 'tan(β) = 15', 'tan(β) = 20', 'tan(β) = 30'],  # Legendas para cada linha
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        xlabel=r"$p_T(q,\bar{{q}})~GeV$",  # Label para o eixo x
        save_name=f"{sets}_qq_pt_tan",  # Nome do arquivo de saída
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 24)],  # Intervalo de pesos para tab(β) = 5
            [f'signal_weight{i}' for i in range(25, 49)],  # Intervalo de pesos para tab(β) = 10
            [f'signal_weight{i}' for i in range(50, 74)],  # Intervalo de pesos para tab(β) = 15
            [f'signal_weight{i}' for i in range(75, 99)],  # Intervalo de pesos para tab(β) = 20
            [f'signal_weight{i}' for i in range(100, 124)],  # Intervalo de pesos para tab(β) = 30
        ]
    )

    # Chamando a função para tan(beta) pT xx
    make_plot(
        dataset_name=sets,  # Nome do dataset
        var="chichi_pt",  # Variável a ser plotada
        bin_vars=bin_pT,  # Intervalo de bins
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Diferentes cores para os intervalos
        labels=['tan(β) = 5', 'tan(β) = 10', 'tan(β) = 15', 'tan(β) = 20', 'tan(β) = 30'],  # Legendas para cada linha
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        xlabel=r"$p_T(\chi,\bar{{\chi}})~GeV$",  # Label para o eixo x
        save_name=f"{sets}_chichi_pt_tan",  # Nome do arquivo de saída
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 24)],  # Intervalo de pesos para tab(β) = 5
            [f'signal_weight{i}' for i in range(25, 49)],  # Intervalo de pesos para tab(β) = 10
            [f'signal_weight{i}' for i in range(50, 74)],  # Intervalo de pesos para tab(β) = 15
            [f'signal_weight{i}' for i in range(75, 99)],  # Intervalo de pesos para tab(β) = 20
            [f'signal_weight{i}' for i in range(100, 124)],  # Intervalo de pesos para tab(β) = 30
        ]
    )

    # Chamando a função para tan(beta) pT qqxx
    make_plot(
        dataset_name=sets,  # Nome do dataset
        var="qqchichi_pt",  # Variável a ser plotada
        bin_vars=bin_pT,  # Intervalo de bins
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Diferentes cores para os intervalos
        labels=['tan(β) = 5', 'tan(β) = 10', 'tan(β) = 15', 'tan(β) = 20', 'tan(β) = 30'],  # Legendas para cada linha
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        xlabel=r"$p_T(q,\overline{q},\chi,\overline{\chi})~GeV$",  # Label para o eixo x
        save_name=f"{sets}_qqchichi_pt_tan",  # Nome do arquivo de saída
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 24)],  # Intervalo de pesos para tab(β) = 5
            [f'signal_weight{i}' for i in range(25, 49)],  # Intervalo de pesos para tab(β) = 10
            [f'signal_weight{i}' for i in range(50, 74)],  # Intervalo de pesos para tab(β) = 15
            [f'signal_weight{i}' for i in range(75, 99)],  # Intervalo de pesos para tab(β) = 20
            [f'signal_weight{i}' for i in range(100, 124)],  # Intervalo de pesos para tab(β) = 30
        ]
    )

    # Chamando a função para sin(Theta) pT qq
    make_plot(
        dataset_name=sets,
        var="qq_pt", 
        bin_vars=bin_pT,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['sin('r'$\theta$) = 0.1', 'sin('r'$\theta$) = 0.3', 'sin('r'$\theta$) = 0.5', 'sin('r'$\theta$) = 0.7', 'sin('r'$\theta$) = 0.9'],  # Rótulos para cada linha
        xlabel=r"$p_T(q,\overline{q})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_qq_pt_sin",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 4)] + [f'signal_weight{i}' for i in range(25, 29)] + [f'signal_weight{i}' for i in range(50, 54)]  + [f'signal_weight{i}' for i in range(75, 79)] + [f'signal_weight{i}' for i in range(100, 104)],  # Sin(theta) = 0.1
            [f'signal_weight{i}' for i in range(5, 9)] + [f'signal_weight{i}' for i in range(30, 34)] + [f'signal_weight{i}' for i in range(55, 59)]  + [f'signal_weight{i}' for i in range(80, 84)] + [f'signal_weight{i}' for i in range(105, 109)],  # Sin(theta) = 0.3
            [f'signal_weight{i}' for i in range(10, 14)] + [f'signal_weight{i}' for i in range(35, 39)] + [f'signal_weight{i}' for i in range(60, 64)]  + [f'signal_weight{i}' for i in range(85, 89)] + [f'signal_weight{i}' for i in range(110, 114)],  # Sin(theta) = 0.5
            [f'signal_weight{i}' for i in range(15, 19)] + [f'signal_weight{i}' for i in range(40, 44)] + [f'signal_weight{i}' for i in range(65, 69)]  + [f'signal_weight{i}' for i in range(90, 94)] + [f'signal_weight{i}' for i in range(115, 119)],  # Sin(theta) = 0.7
            [f'signal_weight{i}' for i in range(20, 24)] + [f'signal_weight{i}' for i in range(45, 49)] + [f'signal_weight{i}' for i in range(70, 74)]  + [f'signal_weight{i}' for i in range(95, 99)] + [f'signal_weight{i}' for i in range(120, 124)],  # Sin(theta) = 0.9
        ]
    )

    # Chamando a função para sin(Theta) pT xx
    make_plot(
        dataset_name=sets,
        var="chichi_pt", 
        bin_vars=bin_pT,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['sin('r'$\theta$) = 0.1', 'sin('r'$\theta$) = 0.3', 'sin('r'$\theta$) = 0.5', 'sin('r'$\theta$) = 0.7', 'sin('r'$\theta$) = 0.9'],  # Rótulos para cada linha
        xlabel=r"$p_T(\chi,\overline{\chi})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_chichi_pt_sin",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 4)] + [f'signal_weight{i}' for i in range(25, 29)] + [f'signal_weight{i}' for i in range(50, 54)]  + [f'signal_weight{i}' for i in range(75, 79)] + [f'signal_weight{i}' for i in range(100, 104)],  # Sin(theta) = 0.1
            [f'signal_weight{i}' for i in range(5, 9)] + [f'signal_weight{i}' for i in range(30, 34)] + [f'signal_weight{i}' for i in range(55, 59)]  + [f'signal_weight{i}' for i in range(80, 84)] + [f'signal_weight{i}' for i in range(105, 109)],  # Sin(theta) = 0.3
            [f'signal_weight{i}' for i in range(10, 14)] + [f'signal_weight{i}' for i in range(35, 39)] + [f'signal_weight{i}' for i in range(60, 64)]  + [f'signal_weight{i}' for i in range(85, 89)] + [f'signal_weight{i}' for i in range(110, 114)],  # Sin(theta) = 0.5
            [f'signal_weight{i}' for i in range(15, 19)] + [f'signal_weight{i}' for i in range(40, 44)] + [f'signal_weight{i}' for i in range(65, 69)]  + [f'signal_weight{i}' for i in range(90, 94)] + [f'signal_weight{i}' for i in range(115, 119)],  # Sin(theta) = 0.7
            [f'signal_weight{i}' for i in range(20, 24)] + [f'signal_weight{i}' for i in range(45, 49)] + [f'signal_weight{i}' for i in range(70, 74)]  + [f'signal_weight{i}' for i in range(95, 99)] + [f'signal_weight{i}' for i in range(120, 124)],  # Sin(theta) = 0.9
        ]
    )

    # Chamando a função para sin(Theta) pT qqxx
    make_plot(
        dataset_name=sets,
        var="qqchichi_pt", 
        bin_vars=bin_pT,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['sin('r'$\theta$) = 0.1', 'sin('r'$\theta$) = 0.3', 'sin('r'$\theta$) = 0.5', 'sin('r'$\theta$) = 0.7', 'sin('r'$\theta$) = 0.9'],  # Rótulos para cada linha
        xlabel=r"$p_T(q,\overline{q},\chi,\overline{\chi})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_qqchichi_pt_sin",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 4)] + [f'signal_weight{i}' for i in range(25, 29)] + [f'signal_weight{i}' for i in range(50, 54)]  + [f'signal_weight{i}' for i in range(75, 79)] + [f'signal_weight{i}' for i in range(100, 104)],  # Sin(theta) = 0.1
            [f'signal_weight{i}' for i in range(5, 9)] + [f'signal_weight{i}' for i in range(30, 34)] + [f'signal_weight{i}' for i in range(55, 59)]  + [f'signal_weight{i}' for i in range(80, 84)] + [f'signal_weight{i}' for i in range(105, 109)],  # Sin(theta) = 0.3
            [f'signal_weight{i}' for i in range(10, 14)] + [f'signal_weight{i}' for i in range(35, 39)] + [f'signal_weight{i}' for i in range(60, 64)]  + [f'signal_weight{i}' for i in range(85, 89)] + [f'signal_weight{i}' for i in range(110, 114)],  # Sin(theta) = 0.5
            [f'signal_weight{i}' for i in range(15, 19)] + [f'signal_weight{i}' for i in range(40, 44)] + [f'signal_weight{i}' for i in range(65, 69)]  + [f'signal_weight{i}' for i in range(90, 94)] + [f'signal_weight{i}' for i in range(115, 119)],  # Sin(theta) = 0.7
            [f'signal_weight{i}' for i in range(20, 24)] + [f'signal_weight{i}' for i in range(45, 49)] + [f'signal_weight{i}' for i in range(70, 74)]  + [f'signal_weight{i}' for i in range(95, 99)] + [f'signal_weight{i}' for i in range(120, 124)],  # Sin(theta) = 0.9
        ]
    )

    # Chamando a função para lan3 pT qq
    make_plot(
        dataset_name=sets,
        var="qq_pt", 
        bin_vars=bin_pT,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['$\lambda_{3} = 0.3$', '$\lambda_{3} = 0.4$', '$\lambda_{3} = 0.5$', '$\lambda_{3} = 0.6$', '$\lambda_{3} = 0.7$'],  # Rótulos para cada linha
        xlabel=r"$p_T(q,\overline{q})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_qq_pt_lambda3",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(125) if i % 5 == 0],  # Múltiplos de 5
            [f'signal_weight{i}' for i in range(125) if i % 5 == 1],  # Múltiplos de 5+1
            [f'signal_weight{i}' for i in range(125) if i % 5 == 2],  # Múltiplos de 5+2
            [f'signal_weight{i}' for i in range(125) if i % 5 == 3],  # Múltiplos de 5+3
            [f'signal_weight{i}' for i in range(125) if i % 5 == 4],  # Múltiplos de 5+4
        ]
    )

    # Chamando a função para lan3 pT xx
    make_plot(
        dataset_name=sets,
        var="chichi_pt", 
        bin_vars=bin_pT,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['$\lambda_{3} = 0.3$', '$\lambda_{3} = 0.4$', '$\lambda_{3} = 0.5$', '$\lambda_{3} = 0.6$', '$\lambda_{3} = 0.7$'],  # Rótulos para cada linha
        xlabel=r"$p_T(\chi,\overline{\chi})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_chichi_pt_lambda3",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(125) if i % 5 == 0],  # Múltiplos de 5
            [f'signal_weight{i}' for i in range(125) if i % 5 == 1],  # Múltiplos de 5+1
            [f'signal_weight{i}' for i in range(125) if i % 5 == 2],  # Múltiplos de 5+2
            [f'signal_weight{i}' for i in range(125) if i % 5 == 3],  # Múltiplos de 5+3
            [f'signal_weight{i}' for i in range(125) if i % 5 == 4],  # Múltiplos de 5+4
        ]
    )

    # Chamando a função para lan3 pT qqxx
    make_plot(
        dataset_name=sets,
        var="qqchichi_pt", 
        bin_vars=bin_pT,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['$\lambda_{3} = 0.3$', '$\lambda_{3} = 0.4$', '$\lambda_{3} = 0.5$', '$\lambda_{3} = 0.6$', '$\lambda_{3} = 0.7$'],  # Rótulos para cada linha
        xlabel=r"$p_T(q,\overline{q},\chi,\overline{\chi})~GeV$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_qqchichi_pt_lambda3",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(125) if i % 5 == 0],  # Múltiplos de 5
            [f'signal_weight{i}' for i in range(125) if i % 5 == 1],  # Múltiplos de 5+1
            [f'signal_weight{i}' for i in range(125) if i % 5 == 2],  # Múltiplos de 5+2
            [f'signal_weight{i}' for i in range(125) if i % 5 == 3],  # Múltiplos de 5+3
            [f'signal_weight{i}' for i in range(125) if i % 5 == 4],  # Múltiplos de 5+4
        ]
    )

if variable == 3 or variable == 0:
    # Chamando a função para tan(beta) Eta qq
    make_plot(
        dataset_name=sets,  # Nome do dataset
        var="qq_eta",  # Variável a ser plotada
        bin_vars=bin_eta,  # Intervalo de bins
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Diferentes cores para os intervalos
        labels=['tan(β) = 5', 'tan(β) = 10', 'tan(β) = 15', 'tan(β) = 20', 'tan(β) = 30'],  # Legendas para cada linha
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        xlabel=r"$\eta(q,\bar{{q}})$",  # Label para o eixo x
        save_name=f"{sets}_eta_tan",  # Nome do arquivo de saída
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 24)],  # Intervalo de pesos para tab(β) = 5
            [f'signal_weight{i}' for i in range(25, 49)],  # Intervalo de pesos para tab(β) = 10
            [f'signal_weight{i}' for i in range(50, 74)],  # Intervalo de pesos para tab(β) = 15
            [f'signal_weight{i}' for i in range(75, 99)],  # Intervalo de pesos para tab(β) = 20
            [f'signal_weight{i}' for i in range(100, 124)],  # Intervalo de pesos para tab(β) = 30
        ]
    )

    # Chamando a função para tan(beta) Eta xx
    make_plot(
        dataset_name=sets,  # Nome do dataset
        var="chichi_eta",  # Variável a ser plotada
        bin_vars=bin_eta,  # Intervalo de bins
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Diferentes cores para os intervalos
        labels=['tan(β) = 5', 'tan(β) = 10', 'tan(β) = 15', 'tan(β) = 20', 'tan(β) = 30'],  # Legendas para cada linha
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        xlabel=r"$\eta(\chi,\bar{{\chi}})$",  # Label para o eixo x
        save_name=f"{sets}_chichi_eta_tan",  # Nome do arquivo de saída
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 24)],  # Intervalo de pesos para tab(β) = 5
            [f'signal_weight{i}' for i in range(25, 49)],  # Intervalo de pesos para tab(β) = 10
            [f'signal_weight{i}' for i in range(50, 74)],  # Intervalo de pesos para tab(β) = 15
            [f'signal_weight{i}' for i in range(75, 99)],  # Intervalo de pesos para tab(β) = 20
            [f'signal_weight{i}' for i in range(100, 124)],  # Intervalo de pesos para tab(β) = 30
        ]
    )


    # Chamando a função para tan(beta) Eta qqxx
    make_plot(
        dataset_name=sets,  # Nome do dataset
        var="qqchichi_eta",  # Variável a ser plotada
        bin_vars=bin_eta,  # Intervalo de bins
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Diferentes cores para os intervalos
        labels=['tan(β) = 5', 'tan(β) = 10', 'tan(β) = 15', 'tan(β) = 20', 'tan(β) = 30'],  # Legendas para cada linha
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        xlabel=r"$\eta(q,\overline{q},\chi,\overline{\chi})$",  # Label para o eixo x
        save_name=f"{sets}_qqchichi_eta_tan",  # Nome do arquivo de saída
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 24)],  # Intervalo de pesos para tab(β) = 5
            [f'signal_weight{i}' for i in range(25, 49)],  # Intervalo de pesos para tab(β) = 10
            [f'signal_weight{i}' for i in range(50, 74)],  # Intervalo de pesos para tab(β) = 15
            [f'signal_weight{i}' for i in range(75, 99)],  # Intervalo de pesos para tab(β) = 20
            [f'signal_weight{i}' for i in range(100, 124)],  # Intervalo de pesos para tab(β) = 30
        ]
    )

    # Chamando a função para sin(Theta) Eta qq
    make_plot(
        dataset_name=sets,
        var="qq_eta", 
        bin_vars=bin_eta,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['sin('r'$\theta$) = 0.1', 'sin('r'$\theta$) = 0.3', 'sin('r'$\theta$) = 0.5', 'sin('r'$\theta$) = 0.7', 'sin('r'$\theta$) = 0.9'],  # Rótulos para cada linha
        xlabel=r"$\eta(q,\overline{q})$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_qq_eta_sin",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 4)] + [f'signal_weight{i}' for i in range(25, 29)] + [f'signal_weight{i}' for i in range(50, 54)] + [f'signal_weight{i}' for i in range(75, 79)] + [f'signal_weight{i}' for i in range(100, 104)],  # Sin(theta) = 0.1
            [f'signal_weight{i}' for i in range(5, 9)] + [f'signal_weight{i}' for i in range(30, 34)] + [f'signal_weight{i}' for i in range(55, 59)] + [f'signal_weight{i}' for i in range(80, 84)] + [f'signal_weight{i}' for i in range(105, 109)],  # Sin(theta) = 0.3
            [f'signal_weight{i}' for i in range(10, 14)] + [f'signal_weight{i}' for i in range(35, 39)] + [f'signal_weight{i}' for i in range(60, 64)] + [f'signal_weight{i}' for i in range(85, 89)] + [f'signal_weight{i}' for i in range(110, 114)],  # Sin(theta) = 0.5
            [f'signal_weight{i}' for i in range(15, 19)] + [f'signal_weight{i}' for i in range(40, 44)] + [f'signal_weight{i}' for i in range(65, 69)] + [f'signal_weight{i}' for i in range(90, 94)] + [f'signal_weight{i}' for i in range(115, 119)],  # Sin(theta) = 0.7
            [f'signal_weight{i}' for i in range(20, 24)] + [f'signal_weight{i}' for i in range(45, 49)] + [f'signal_weight{i}' for i in range(70, 74)] + [f'signal_weight{i}' for i in range(95, 99)] + [f'signal_weight{i}' for i in range(120, 124)],  # Sin(theta) = 0.9
        ]
    )

    # Chamando a função para sin(Theta) Eta xx
    make_plot(
        dataset_name=sets,
        var="chichi_eta", 
        bin_vars=bin_eta,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['sin('r'$\theta$) = 0.1', 'sin('r'$\theta$) = 0.3', 'sin('r'$\theta$) = 0.5', 'sin('r'$\theta$) = 0.7', 'sin('r'$\theta$) = 0.9'],  # Rótulos para cada linha
        xlabel=r"$\eta(\chi,\overline{\chi})$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_chichi_eta_sin",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 4)] + [f'signal_weight{i}' for i in range(25, 29)] + [f'signal_weight{i}' for i in range(50, 54)] + [f'signal_weight{i}' for i in range(75, 79)] + [f'signal_weight{i}' for i in range(100, 104)],  # Sin(theta) = 0.1
            [f'signal_weight{i}' for i in range(5, 9)] + [f'signal_weight{i}' for i in range(30, 34)] + [f'signal_weight{i}' for i in range(55, 59)] + [f'signal_weight{i}' for i in range(80, 84)] + [f'signal_weight{i}' for i in range(105, 109)],  # Sin(theta) = 0.3
            [f'signal_weight{i}' for i in range(10, 14)] + [f'signal_weight{i}' for i in range(35, 39)] + [f'signal_weight{i}' for i in range(60, 64)] + [f'signal_weight{i}' for i in range(85, 89)] + [f'signal_weight{i}' for i in range(110, 114)],  # Sin(theta) = 0.5
            [f'signal_weight{i}' for i in range(15, 19)] + [f'signal_weight{i}' for i in range(40, 44)] + [f'signal_weight{i}' for i in range(65, 69)] + [f'signal_weight{i}' for i in range(90, 94)] + [f'signal_weight{i}' for i in range(115, 119)],  # Sin(theta) = 0.7
            [f'signal_weight{i}' for i in range(20, 24)] + [f'signal_weight{i}' for i in range(45, 49)] + [f'signal_weight{i}' for i in range(70, 74)] + [f'signal_weight{i}' for i in range(95, 99)] + [f'signal_weight{i}' for i in range(120, 124)],  # Sin(theta) = 0.9
        ]
    )


    # Chamando a função para sin(Theta) Eta qqxx
    make_plot(
        dataset_name=sets,
        var="qqchichi_eta", 
        bin_vars=bin_eta,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['sin('r'$\theta$) = 0.1', 'sin('r'$\theta$) = 0.3', 'sin('r'$\theta$) = 0.5', 'sin('r'$\theta$) = 0.7', 'sin('r'$\theta$) = 0.9'],  # Rótulos para cada linha
        xlabel=r"$\eta(q,\overline{q},\chi,\overline{\chi})$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_qqchichi_eta_sin",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(0, 4)] + [f'signal_weight{i}' for i in range(25, 29)] + [f'signal_weight{i}' for i in range(50, 54)] + [f'signal_weight{i}' for i in range(75, 79)] + [f'signal_weight{i}' for i in range(100, 104)],  # Sin(theta) = 0.1
            [f'signal_weight{i}' for i in range(5, 9)] + [f'signal_weight{i}' for i in range(30, 34)] + [f'signal_weight{i}' for i in range(55, 59)] + [f'signal_weight{i}' for i in range(80, 84)] + [f'signal_weight{i}' for i in range(105, 109)],  # Sin(theta) = 0.3
            [f'signal_weight{i}' for i in range(10, 14)] + [f'signal_weight{i}' for i in range(35, 39)] + [f'signal_weight{i}' for i in range(60, 64)] + [f'signal_weight{i}' for i in range(85, 89)] + [f'signal_weight{i}' for i in range(110, 114)],  # Sin(theta) = 0.5
            [f'signal_weight{i}' for i in range(15, 19)] + [f'signal_weight{i}' for i in range(40, 44)] + [f'signal_weight{i}' for i in range(65, 69)] + [f'signal_weight{i}' for i in range(90, 94)] + [f'signal_weight{i}' for i in range(115, 119)],  # Sin(theta) = 0.7
            [f'signal_weight{i}' for i in range(20, 24)] + [f'signal_weight{i}' for i in range(45, 49)] + [f'signal_weight{i}' for i in range(70, 74)] + [f'signal_weight{i}' for i in range(95, 99)] + [f'signal_weight{i}' for i in range(120, 124)],  # Sin(theta) = 0.9
        ]
    )

    # Chamando a função para lambda3 Eta qq
    make_plot(
        dataset_name=sets,
        var="qq_eta", 
        bin_vars=bin_eta,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['$\lambda_{3} = 0.3$', '$\lambda_{3} = 0.4$', '$\lambda_{3} = 0.5$', '$\lambda_{3} = 0.6$', '$\lambda_{3} = 0.7$'],  # Rótulos para cada linha
        xlabel=r"$\eta(q,\overline{q})$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_qq_eta_lambda3",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(125) if i % 5 == 0],  # Múltiplos de 5
            [f'signal_weight{i}' for i in range(125) if i % 5 == 1],  # Múltiplos de 5+1
            [f'signal_weight{i}' for i in range(125) if i % 5 == 2],  # Múltiplos de 5+2
            [f'signal_weight{i}' for i in range(125) if i % 5 == 3],  # Múltiplos de 5+3
            [f'signal_weight{i}' for i in range(125) if i % 5 == 4],  # Múltiplos de 5+4
        ]
    )

    # Chamando a função para lambda3 Eta xx
    make_plot(
        dataset_name=sets,
        var="chichi_eta", 
        bin_vars=bin_eta,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['$\lambda_{3} = 0.3$', '$\lambda_{3} = 0.4$', '$\lambda_{3} = 0.5$', '$\lambda_{3} = 0.6$', '$\lambda_{3} = 0.7$'],  # Rótulos para cada linha
        xlabel=r"$\eta(\chi,\overline{\chi})$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_chichi_eta_lambda3",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(125) if i % 5 == 0],  # Múltiplos de 5
            [f'signal_weight{i}' for i in range(125) if i % 5 == 1],  # Múltiplos de 5+1
            [f'signal_weight{i}' for i in range(125) if i % 5 == 2],  # Múltiplos de 5+2
            [f'signal_weight{i}' for i in range(125) if i % 5 == 3],  # Múltiplos de 5+3
            [f'signal_weight{i}' for i in range(125) if i % 5 == 4],  # Múltiplos de 5+4
        ]
    )

    make_plot(
        dataset_name=sets,
        var="qqchichi_eta", 
        bin_vars=bin_eta,
        colors=['red', 'green', 'blue', 'yellow', 'purple'],  # Cores para cada grupo
        labels=['$\lambda_{3} = 0.3$', '$\lambda_{3} = 0.4$', '$\lambda_{3} = 0.5$', '$\lambda_{3} = 0.6$', '$\lambda_{3} = 0.7$'],  # Rótulos para cada linha
        xlabel=r"$\eta(q,\overline{q},\chi,\overline{\chi})$",
        line_styles=['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))],  # Estilos das linhas
        normalize=True,  # Normaliza pela área
        save_name=f"{sets}_qqchichi_eta_lambda3",
        custom_weight_groups=[
            [f'signal_weight{i}' for i in range(125) if i % 5 == 0],  # Múltiplos de 5
            [f'signal_weight{i}' for i in range(125) if i % 5 == 1],  # Múltiplos de 5+1
            [f'signal_weight{i}' for i in range(125) if i % 5 == 2],  # Múltiplos de 5+2
            [f'signal_weight{i}' for i in range(125) if i % 5 == 3],  # Múltiplos de 5+3
            [f'signal_weight{i}' for i in range(125) if i % 5 == 4],  # Múltiplos de 5+4
        ]
    )
