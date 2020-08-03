#!/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python 
import ROOT as rt
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import gc


def muno_info(pos_theta, pos_phi, pos_r, theta_mom, phi_mom, energy):
    lowX=0.1
    lowY=0.8
    info  = rt.TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    info.SetBorderSize(   0 )
    info.SetFillStyle(    0 )
    info.SetTextAlign(   12 )
    info.SetTextColor(    1 )
    info.SetTextSize(0.03)
    info.SetTextFont (   42 )
    info.AddText("#mu(pos_{#theta}^{%.2f}, pos_{#phi}^{%.2f}, pos_{r}^{%.1f}, mom_{#theta}^{%.2f}, mom_{#phi}^{%.2f}, E=%d MeV)"%(pos_theta, pos_phi, pos_r, theta_mom, phi_mom, energy))

def do_plot(muon,hist,out_name,title):
    canvas=rt.TCanvas("can_%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #h_corr.Draw("COLZ")
    #h_corr.LabelsDeflate("X")
    #h_corr.LabelsDeflate("Y")
    #h_corr.LabelsOption("v")
    hist.SetStats(rt.kFALSE)
    hist.GetXaxis().SetTitle("#phi(AU)")
    hist.GetYaxis().SetTitle("Z(AU)")
    hist.SetTitle(title)
    #hist.SetTitleSize(0.1)
    #hist.Draw("COLZ TEXT")
    hist.Draw("COLZ")
    if muon != None:
        Info = muno_info(muon[2], muon[3], muon[4], muon[0], muon[1], muon[5])
        Info.Draw()
    canvas.SaveAs("%s/%s.pdf"%(path_plots, out_name))
    del canvas
    gc.collect()


def plot_image(image, name, title, muon_info=None):
    if muon_info != None : print ('muon Direction: theta=%f, phi=%f. Position: theta=%f, phi=%f, r=%f. Energy=%f MeV'%( muon_info[0], muon_info[1], muon_info[2], muon_info[3], muon_info[4], muon_info[5]))
    nX = image.shape[1]
    nY = image.shape[0]

    h = rt.TH2F('%s'%name, '', nX, 0, nX, nY, 0, nY)
    for i in range(0, nX):
        i0 = nX-1 - i
        for j in range(0, nY):
            j0 = nY-1 - j
            h.Fill(i0+0.01, j0+0.01, image[j][i])
    do_plot(muon_info, h, name , title)



#matplotlib.rcParams.update({'font.size': 20})
def plot_img(img, name, vmin=None, vmax=None):## can't run it when submit the job to cluster, use plot_image() then
    fig = plt.figure(figsize=(20,10))
    plt.imshow(img) # ÊòæÁ§∫ÂõæÁâá
    plt.axis('on') # ÊòæÁ§∫ÂùêÊ†áËΩ¥
    plt.xlabel(r'$\phi$ PMT')
    plt.ylabel(r'Z PMT')
    cbar = plt.colorbar()
    cbar.set_label(r'PMT_%s'%name, y=0.99)
    plt.tight_layout()
    plt.savefig('%s/%s.pdf'%(path_plots,name))
    plt.close(fig)
    del fig
    gc.collect()

path_plots = '/hpcfs/juno/junogpu/fangwx/FastSim/analysis/plots/'
d_muon = h5py.File('/hpcfs/juno/junogpu/fangwx/FastSim/data/data_all0.h5', 'r')

muon_0 = d_muon['firstHitTimeByPMT'][:]
muon_1 = d_muon['nPEByPMT'][:]
muon_2 = d_muon['infoMuon'][:]

# let's look at some plots
'''
for i in [0,1]:
    plot_img(muon_0[i], 'hit_time_evt%d'%i)
    plot_img(muon_1[i], 'nPE_evt%d'%i)
'''
for i in [0,1]:
    plot_image(muon_0[i], 'hit_time_evt%d'%i, 'first hit of PMTs', muon_2[i])
    plot_image(muon_1[i], 'nPE_evt%d'%i     , 'nPE of PMTs'      , muon_2[i])

#sys.exit()
#Let's build the network architecture we proposed, load its trained weights, and use it to generate synthesized showers. We do all this using Keras with the TensorFlow backend


sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/models/')
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/analysis/')
from keras.layers import Input, Lambda, Activation, AveragePooling2D, UpSampling2D
from keras.models import Model
from keras.layers.merge import multiply
import keras.backend as K

from architectures import build_generator, build_discriminator, sparse_softmax
from ops import scale, inpainting_attention

#This was our choice for the size of the latent space  Ì†µÌ±ß . If you want to retrain this net, you can try changing this parameter.
latent_size = 256
latent = Input(shape=(latent_size, ), name='z') 

generator_inputs = [latent]

img_layer0 = build_generator(latent, 126, 226)
img_layer1 = build_generator(latent, 126, 226)

generator_outputs = [
    Activation('relu')(img_layer0),
    Activation('relu')(img_layer1)
]
# build the actual model
generator = Model(generator_inputs, generator_outputs)
#print(generator.summary())

# load trained weights
generator.load_weights('/hpcfs/juno/junogpu/fangwx/FastSim/params_generator_epoch_049.hdf5')

n_gen_images = 10

noise = np.random.normal(0, 1, (n_gen_images, latent_size))

images = generator.predict([noise], verbose=True)

print (images[0].shape)
print (images[1].shape)

images = np.squeeze(images)

for i in range(0, 3):
    #plot_img(images[0][i,:], 'gen_hit_time_evt%d'%i)
    #plot_img(images[1][i,:], 'gen_nPE_evt%d'%i)
    plot_image(images[0][i,:], 'gen_hit_time_evt%d'%i, 'first hit of PMTs', None)
    plot_image(images[1][i,:], 'gen_nPE_evt%d'%i     , 'nPE of PMTs'      , None)

print ('done!')
