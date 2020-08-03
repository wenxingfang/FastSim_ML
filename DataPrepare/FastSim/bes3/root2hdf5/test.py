import ROOT as rt
import numpy as np

if True:

    f_out = rt.TFile('test_clone.root','RECREATE')

    h = rt.TH1F('test','',20,0,20)

    for i in range(10):
        h.SetBinContent(i+1, 2)


    bins=np.array([0, 1, 4, 6, 10, 20], dtype=np.float32)
    hh = rt.TH1F('del','', 5, bins)
    hh.SetBinContent(1, 30)
    hh.SetBinContent(4, 30)

    f_out.cd()
    h.Write()
    hh.Write()
    f_out.Close()
