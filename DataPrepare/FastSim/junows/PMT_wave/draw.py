import ROOT as rt
import gc
rt.gROOT.SetBatch(rt.kTRUE)

def plot_gr(gr,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    '''
    x1 = 45
    x2 = 135
    y1 = x1
    y2 = x2
    if 'theta' in out_name:
        gr.GetXaxis().SetTitle("True #theta")
        gr.GetYaxis().SetTitle("Predicted #theta")
        x1 = 45
        x2 = 135
        y1 = x1
        y2 = x2
    elif 'phi' in out_name:
        gr.GetXaxis().SetTitle("True #phi")
        gr.GetYaxis().SetTitle("Predicted #phi")
        x1 = -10
        x2 = 10
        y1 = x1
        y2 = x2
    elif 'energy' in out_name:
        gr.GetXaxis().SetTitle("True Energy")
        gr.GetYaxis().SetTitle("Predicted Energy")
        x1 = 10
        x2 = 100
        y1 = x1
        y2 = x2
    #gr.SetTitle(title)
    gr.Draw("ap")
    
    line = rt.TLine(x1, y1, x2, y2)
    line.SetLineColor(rt.kRed)
    line.SetLineWidth(2)
    line.Draw('same')
    '''
    gr.Draw("ap")
    
    canvas.SaveAs("%s.png"%(out_name))
    del canvas
    gc.collect()

f = open('/junofs/users/wxfang/FastSim/junows/PMT_wave/build/test_part2.txt')
i=0
gr =  rt.TGraph()
gr.SetMarkerStyle(8)
for p in f:
    p=float(p)    
    if p > 100 or p < -100: continue
    gr.SetPoint(i,i, p)
    i=i+1
    if i > 600: break
plot_gr(gr, "pmt","")
