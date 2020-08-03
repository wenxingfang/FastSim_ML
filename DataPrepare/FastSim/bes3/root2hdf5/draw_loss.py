import ROOT as rt
import gc
rt.gROOT.SetBatch(rt.kTRUE)


def plot_gr(gr, gr1, gr2, gr3, out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetGridy()
    canvas.SetGridx()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    dummy = rt.TH2D("dummy","",1,0,200,1,-5,10)
    dummy.SetStats(rt.kFALSE)
    dummy_Y_title="Loss"
    dummy_X_title="Epoch"
    dummy.GetYaxis().SetTitle(dummy_Y_title)
    dummy.GetXaxis().SetTitle(dummy_X_title)
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.7)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw()
    gr.Draw ("p")
    gr1.Draw('p')
    gr2.Draw('p')
    gr3.Draw('p')
    dummy.Draw("AXISSAME")
    legend = rt.TLegend(0.6,0.6,0.85,0.85)
    #legend.AddEntry(h_real,'Data','lep')
    legend.AddEntry(gr,'total','p')
    legend.AddEntry(gr1,"real",'p')
    legend.AddEntry(gr2,"fake",'p')
    legend.AddEntry(gr3,"average",'p')
    legend.SetBorderSize(0)
    #legend.SetLineColor(1)
    #legend.SetLineStyle(1)
    #legend.SetLineWidth(1)
    #legend.SetFillColor(19)
    #legend.SetFillStyle(0)
    #legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s.png"%(out_name))
    del canvas
    gc.collect()


#file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/wgan_out_1110.txt'
file_in = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/params/gan/wgan_out_1125.txt'

f_in = open(file_in, 'r')
lines = f_in.readlines()

gr_loss_total     =  rt.TGraph()
gr_loss_real      =  rt.TGraph()
gr_loss_fake      =  rt.TGraph()
gr_loss_avge      =  rt.TGraph()

gr_loss_total.SetMarkerColor(rt.kBlack)
gr_loss_total.SetMarkerStyle(8)
gr_loss_real .SetMarkerColor(rt.kRed)
gr_loss_real .SetMarkerStyle(8)
gr_loss_fake .SetMarkerColor(rt.kBlue)
gr_loss_fake .SetMarkerStyle(8)
gr_loss_avge .SetMarkerColor(rt.kGreen)
gr_loss_avge .SetMarkerStyle(8)

i = 0
for line in lines:
    value = line.split(' ')
    
    gr_loss_total.SetPoint(i, float(value[0]), float(value[1]))
    gr_loss_real .SetPoint(i, float(value[0]), float(value[2]))
    gr_loss_fake .SetPoint(i, float(value[0]), float(value[3]))
    gr_loss_avge .SetPoint(i, float(value[0]), float(value[4]))
    i = i + 1

plot_gr(gr_loss_total, gr_loss_real, gr_loss_fake, gr_loss_avge, "gr_loss_total","")
