import plotly.graph_objects as go
import numpy as np
import scipy.io as sio
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px

from data_load_norm import dataimport2D, labelsimport, labelsNorm, ilabelsNorm, inputConcat2D, dataimport2D_md, labelsimport_md, dataimport2Dhres

# Import datasets
folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'

data_import = sio.loadmat(folder + 'dataset_spectra_TEST.mat')
conc_import = sio.loadmat(folder + 'labels_c_TEST_abs.mat')
snr_v = sio.loadmat(folder + 'snr_v_TEST')
metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']

nset= 5
data = np.zeros((2500,1024))
snr = np.zeros((2500,1))
conc = np.zeros((17,2500))
for k in range(2500):
    data[k,:] = data_import['dataset_spectra'][k][0][:]
    snr[k,:] = snr_v['snr_v'][k]
    conc[:,k] = conc_import['labels_c'][:,k]*64.5
# fig = go.Figure(go.Scatter(y=data[0,:], mode='lines'))

conc = np.transpose(conc,(1,0))
snr = snr[:,0]

# sorting on abs SNR
snr_sort_arg = np.argsort(snr,axis=0)
data_sort = data[snr_sort_arg,:]
snr_sort = snr[snr_sort_arg]
conc_sort = conc[snr_sort_arg,:]
data_sort_group = np.reshape(data_sort, (nset, int(2500/nset), 1024))
snr_sort_group = np.reshape(snr_sort, (nset, int(2500/nset)))
conc_sort_group = np.reshape(conc_sort, (nset, int(2500/nset), 17))
# # sorting on rel SNR (NAA)
# rsnr = np.multiply(snr,conc[:,2])
# rsnr_sort_arg = np.argsort(rsnr,axis=0)
# rdata_sort = data[rsnr_sort_arg,:]
# rsnr_sort = rsnr[rsnr_sort_arg]
# rdata_sort_group = np.reshape(rdata_sort, (25, 100, 1024))
# rsnr_sort_group = np.reshape(rsnr_sort, (25, 100))


# Create figure
fig = make_subplots(rows=2, cols=2)

# Add traces, one for each slider step
for step in range(nset):
        fig.add_trace(
            go.Scatter(
                visible=False,
                # line=dict(color="#00CED1", width=6),
                # name="𝜈 = " + str(step),
                # y=np.mean(data_sort_group[step,:,:], axis=0)))
                y = data_sort_group[step, 0, :]),
                row=1, col=1)
for step in range(nset):
    hist_data = [conc_sort_group[step, :, 2]]
    fig1 = ff.create_distplot(hist_data=hist_data, group_labels=['distplot'])
    fig.add_trace(
        # go.Histogram(dict(enumerate(conc_sort_group[step,:,2].flatten(), 1), )))
        # go.Histogram(x=conc_sort_group[step, :, 2], nbinsx=100),
        go.Scatter(
            fig1['data'][1],
            visible=False),
            row=1, col=2)

    fig.add_trace(
        go.Histogram(
            fig1['data'][0],
            visible = False),
            row=1, col=2)
        # fig.add_trace(
        #     go.Scatter(
        #         visible=False,
        #         # line=dict(color="#00CED1", width=6),
        #         # name="𝜈 = " + str(step),
        #         y=np.mean(data_sort_group[step, :, :], axis=0) + np.std(data_sort_group[step, :, :], axis=0)))
        # fig.add_trace(
        #     go.Scatter(
        #         visible=False,
        #         # line=dict(color="#00CED1", width=6),
        #         # name="𝜈 = " + str(step),
        #         y=np.mean(data_sort_group[step, :, :], axis=0) - np.std(data_sort_group[step, :, :], axis=0)))

# Make 10th trace visible
fig.data[int(nset/2)].visible = True

# Create and add slider
steps = []
for i in range(nset):
    step = dict(
        method="update",
        args=[{"visible": [False] * nset},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=int(nset/2),
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 1},
    steps=steps
)]

fig.update_yaxes(title_text="yaxis 2 title", range=[-5000, 45000], row=1, col=1)

fig.update_layout(
    sliders=sliders,
    # yaxis_range=[-5000,45000]
)
fig.write_html("figure_1.html", auto_open=True)
