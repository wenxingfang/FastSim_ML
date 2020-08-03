import ROOT as rt
from array import array
import gc
import math

import os
rt.gSystem.Load("libRooFit")
rt.gSystem.Load("RooDCBShape_cxx.so") #This is the way to handle user defined pdf (generated with RooClassFactory). Compile once and for all in rt and then add it as a library
rt.gROOT.SetBatch(rt.kTRUE)
rt.TH1.AddDirectory(rt.kFALSE)
rt.RooAbsReal.defaultIntegratorConfig().setEpsAbs(1e-10) 
rt.RooAbsReal.defaultIntegratorConfig().setEpsRel(1e-10) 


def fit_plot(dir_name, fit_range, out_name,hist,Dict, doChi2, pdf_index):

    m         = rt.RooRealVar("m"       ,""                      ,fit_range[0], fit_range[1])
    # RooCB, pdf_index=2
    CB_mean   = rt.RooRealVar('mean'     , '', 0.006,-0.1,0.1)
    CB_sigma  = rt.RooRealVar('sigma'    , '', 0.007, 0.,1)
    CB_alpha  = rt.RooRealVar('alpha'    , '', 10.0 , 0 ,100)
    CB_n      = rt.RooRealVar('n'        , '', 0.5  , 0 ,10)
    CB        = rt.RooCBShape('CB'       , 'Cystal Ball Function', m, CB_mean, CB_sigma, CB_alpha, CB_n)
    # RooDCB, pdf_index=3
    dCBmean   = rt.RooRealVar("mean"     , "DCB Bias"       , Dict["dcb"].mean     [0],  Dict["dcb"].mean     [1], Dict["dcb"].mean     [2])
    dCBsigma  = rt.RooRealVar("sigma"    , "DCB Width"      , Dict["dcb"].sigma    [0],  Dict["dcb"].sigma    [1], Dict["dcb"].sigma    [2])
    dCBCutL   = rt.RooRealVar("CutL"     , "DCB Cut left"   , Dict["dcb"].dCBCutL  [0],  Dict["dcb"].dCBCutL  [1], Dict["dcb"].dCBCutL  [2])
    dCBCutR   = rt.RooRealVar("CutR"     , "DCB Cut right"  , Dict["dcb"].dCBCutR  [0],  Dict["dcb"].dCBCutR  [1], Dict["dcb"].dCBCutR  [2])
    dCBPowerL = rt.RooRealVar("PowerL"   , "DCB Power left" , Dict["dcb"].dCBPowerL[0],  Dict["dcb"].dCBPowerL[1], Dict["dcb"].dCBPowerL[2])
    dCBPowerR = rt.RooRealVar("PowerR"   , "DCB Power right", Dict["dcb"].dCBPowerR[0],  Dict["dcb"].dCBPowerR[1], Dict["dcb"].dCBPowerR[2])
    dcb       = rt.RooDCBShape("dcb"     , "double crystal ball", m, dCBmean, dCBsigma, dCBCutL, dCBCutR, dCBPowerL, dCBPowerR)

    Hist      =rt.RooDataHist("Hist","",rt.RooArgList(m), hist)
    frame = m.frame(rt.RooFit.Name(""),rt.RooFit.Title(" "))
    frame.SetMaximum(2*hist.GetMaximum())
    
    Hist.plotOn(frame) 
    result=rt.RooFitResult()
    model=rt.RooAddPdf()
    if pdf_index==1:
        pass
    elif pdf_index==2:
        model=CB
    elif pdf_index==3:
        model=dcb
    else:
        print ("wrong pdf index")
        exit()

    if doChi2==False:
        #result = model.fitTo(Hist, rt.RooFit.Range(-0.8, 50e-3), rt.RooFit.Save(True))
        result = model.fitTo(Hist, rt.RooFit.Save(True))
    else:
        cmd = rt.RooFit.Save()
        currentlist = rt.RooLinkedList()
        currentlist.Add(cmd)
        result = model.chi2FitTo(Hist, currentlist)
    fit_status=rt.gMinuit.fCstatu
    model.plotOn(frame,rt.RooFit.LineStyle(1),rt.RooFit.LineColor(rt.kBlue))
    model.paramOn(frame, rt.RooFit.Layout(0.11,0.35,0.9))##plot parameter
    frame.getAttText().SetTextSize(0.015)

    c = rt.TCanvas("fit","fit",800,800) #X length, Y length
    c.cd()
    #c.SetLogy()
    c.SetTickx(1)
    c.SetTicky(1)
    hist.Draw('pe')

    frame.SetMinimum(1) 
      
    frame.SetYTitle("Events / %.1f"%(hist.GetBinWidth(1))) # change y title
    frame.SetTitleOffset(1.5,"Y")#
    frame.SetLabelSize(0.025,"Y")#
    frame.SetXTitle("(M_{reco}-M_{gen})/M_{gen}") # change x title
    frame.SetTitleOffset(1.2,"X")#
    #frame.Draw() 
    frame.Draw('SAME') 
    cms =rt.TLatex(0.15,0.91,"CEPC Internal")
    cms.SetNDC()
    cms.SetTextSize(0.04)
    cms.Draw()
    str_region=hist.GetName()
    region =rt.TLatex(0.65,0.86,"#color[2]{%s}"%(str_region))
    region.SetNDC()
    region.SetTextSize(0.04)
    region.Draw()
    
    tText = rt.TPaveText(0.55, 0.75, 0.65, 0.85, "brNDC")
    tText.SetBorderSize(0)
    tText.SetFillColor(0)
    tText.SetFillStyle(0)
    tText.SetTextSize(0.032);
#    t1 = tText.AddText("mean   = %.4f #pm %.4f"%(mean_fit  ,mean_fit_error  ))
#    t2 = tText.AddText("sigmaL = %.4f #pm %.4f"%(sigmaL_fit,sigmaL_fit_error))
#    t2 = tText.AddText("sigmaR = %.4f #pm %.4f"%(sigmaR_fit,sigmaR_fit_error))
#    t2 = tText.AddText("alphaL = %.4f #pm %.4f"%(alphaL_fit,alphaL_fit_error))
#    t2 = tText.AddText("alphaR = %.4f #pm %.4f"%(alphaR_fit,alphaR_fit_error))
    if Print_status:
        t3 = tText.AddText("status = %s"%(fit_status))
    tText.Draw()
    c.SaveAs(str(dir_name+'/'+'/'+out_name+'.png'))
    out_dict={}
    for ip in range(0,result.floatParsFinal().getSize()):
        tmp_para=result.floatParsFinal().at(ip)
        tmp_name=tmp_para.GetName()
        tmp_value=tmp_para.getVal()
        tmp_error=tmp_para.getError()
        out_dict[tmp_name]=[tmp_value,tmp_error]
    print (out_dict)
    return out_dict


def do_plot(hist,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
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
    ystr = ''
    if 'GeV/c^{2}' in title['X']:
        ystr = 'GeV/c^{2}'
    elif 'GeV/c' in title['X']:
        ystr = 'GeV/c'
    elif 'GeV' in title['X']:
        ystr = 'GeV'
    hist.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(hist.GetBinWidth(1), ystr)))
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.SetLineColor(rt.kBlue)
    hist.SetMarkerColor(rt.kBlue)
    hist.SetMarkerStyle(20)
    hist.Draw("histep")
    func = hist.GetFunction("f1")
    func.Draw('SAME')
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

class Para_DCB:
    def __init__(self,name,mean,sigma,dCBCutL,dCBCutR,dCBPowerL,dCBPowerR):
        self.name=name
        self.mean=mean
        self.sigma=sigma
        self.dCBCutL=dCBCutL
        self.dCBCutR=dCBCutR
        self.dCBPowerL=dCBPowerL
        self.dCBPowerR=dCBPowerR

def plot_gr(hist,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    hist.GetYaxis().SetTitle(title['Y'])
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.SetLineColor(rt.kBlue)
    hist.SetMarkerColor(rt.kBlue)
    hist.SetMarkerStyle(20)
    hist.Draw("ap")
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

if __name__ == '__main__':
 
    Print_status = True
 
    plot_path = './plots_fit'
    f_in= rt.TFile('out.root','read')
    para_dict = {}
    for ih in range(0,f_in.GetListOfKeys().GetSize()):
        hname=f_in.GetListOfKeys()[ih].GetName()
        hist = f_in.Get(hname)
        mean = hist.GetMean()
        rms = hist.GetRMS()
        lower = 2.5*rms
        higher = 2.5*rms
        #lower = 0.15
        #higher = 0.15
        ''' 
        f1 = rt.TF1("f1", "gaus", mean-lower, mean+higher)
        f1.SetParameters(0.01, 0.1)
        result = hist.Fit('f1','RLS')
        par0   = result.Parameter(0)
        err0   = result.ParError(0)
        par1   = result.Parameter(1)
        err1   = result.ParError(1)
        par2   = result.Parameter(2)
        err2   = result.ParError (2)
        status = result.Status()
        print('%s:mean=%f,mean err=%f,sigma=%f, sigma err=%f, status=%d'%(hname, par1, err1, par2, err2, status))
        do_plot(hist, 'fit_%s'%hname, {'X':'','Y':'Events'})
        para_dict[hname] = [par1, err1, par2, err2]
        ''' 
        init_fit_para ={"dcb":Para_DCB (name="dcb",mean=[0.,-0.1,0.1],sigma=[0.02, 0.,0.1],dCBCutL=[0.05,0.,1],dCBCutR=[0.05,0.,1],dCBPowerL=[1.0,0.,20],dCBPowerR=[1.0,0.,50])}
        fit_result= fit_plot(dir_name=plot_path, fit_range=[mean-lower, mean+higher], out_name='fit_%s'%hname, hist=hist, Dict=init_fit_para, doChi2=False, pdf_index=3 )
    f_in.Close()
    '''
    gr_E_x           = array('f')
    gr_E_x_low       = array('f')
    gr_E_x_high      = array('f')
    gr_E_y0          = array('f')
    gr_E_y0_low      = array('f')
    gr_E_y0_high     = array('f')
    gr_E_y1          = array('f')
    gr_E_y1_low      = array('f')
    gr_E_y1_high     = array('f')
    gr_Theta_x       = array('f')
    gr_Theta_x_low   = array('f')
    gr_Theta_x_high  = array('f')
    gr_Theta_y0      = array('f')
    gr_Theta_y0_low  = array('f')
    gr_Theta_y0_high = array('f')
    gr_Theta_y1      = array('f')
    gr_Theta_y1_low  = array('f')
    gr_Theta_y1_high = array('f')
    gr_Phi_x         = array('f')
    gr_Phi_x_low     = array('f')
    gr_Phi_x_high    = array('f')
    gr_Phi_y0        = array('f')
    gr_Phi_y0_low    = array('f')
    gr_Phi_y0_high   = array('f')
    gr_Phi_y1        = array('f')
    gr_Phi_y1_low    = array('f')
    gr_Phi_y1_high   = array('f')
    for name in para_dict:
        if 'E_' in name:
            low = float(name.split('_')[1])
            high = float(name.split('_')[-1])
            mean = (low + high)/2
            gr_E_x.append(mean) 
            gr_E_x_low.append(mean-low) 
            gr_E_x_high.append(high-mean) 
            gr_E_y0.append(para_dict[name][0]) 
            gr_E_y0_low.append(para_dict[name][1]) 
            gr_E_y0_high.append(para_dict[name][1]) 
            gr_E_y1.append(para_dict[name][2]) 
            gr_E_y1_low.append(para_dict[name] [3]) 
            gr_E_y1_high.append(para_dict[name][3]) 
        elif 'Theta_' in name:
            low  = float(name.split('_')[1])
            high = float(name.split('_')[-1])
            mean = (low + high)/2
            gr_Theta_x.append(mean) 
            gr_Theta_x_low  .append(mean-low) 
            gr_Theta_x_high .append(high-mean) 
            gr_Theta_y0     .append(para_dict[name][0]) 
            gr_Theta_y0_low .append(para_dict[name][1]) 
            gr_Theta_y0_high.append(para_dict[name][1]) 
            gr_Theta_y1     .append(para_dict[name][2]) 
            gr_Theta_y1_low .append(para_dict[name][3]) 
            gr_Theta_y1_high.append(para_dict[name][3]) 
        elif 'Phi_' in name:
            low  = float(name.split('_')[1])
            high = float(name.split('_')[-1])
            mean = (low + high)/2
            gr_Phi_x.append(mean) 
            gr_Phi_x_low  .append(mean-low) 
            gr_Phi_x_high .append(high-mean) 
            gr_Phi_y0     .append(para_dict[name][0]) 
            gr_Phi_y0_low .append(para_dict[name][1]) 
            gr_Phi_y0_high.append(para_dict[name][1]) 
            gr_Phi_y1     .append(para_dict[name][2]) 
            gr_Phi_y1_low .append(para_dict[name][3]) 
            gr_Phi_y1_high.append(para_dict[name][3]) 
           
    gr_E_mean = rt.TGraphAsymmErrors(int(len(gr_E_x)),gr_E_x, gr_E_y0, gr_E_x_low,gr_E_x_high, gr_E_y0_low, gr_E_y0_high) 
    gr_E_std  = rt.TGraphAsymmErrors(int(len(gr_E_x)),gr_E_x, gr_E_y1, gr_E_x_low,gr_E_x_high, gr_E_y1_low, gr_E_y1_high) 
    gr_Theta_mean = rt.TGraphAsymmErrors(int(len(gr_Theta_x)),gr_Theta_x, gr_Theta_y0, gr_Theta_x_low,gr_Theta_x_high, gr_Theta_y0_low, gr_Theta_y0_high) 
    gr_Theta_std  = rt.TGraphAsymmErrors(int(len(gr_Theta_x)),gr_Theta_x, gr_Theta_y1, gr_Theta_x_low,gr_Theta_x_high, gr_Theta_y1_low, gr_Theta_y1_high) 
    gr_Phi_mean = rt.TGraphAsymmErrors(int(len(gr_Phi_x)),gr_Phi_x, gr_Phi_y0, gr_Phi_x_low,gr_Phi_x_high, gr_Phi_y0_low, gr_Phi_y0_high) 
    gr_Phi_std  = rt.TGraphAsymmErrors(int(len(gr_Phi_x)),gr_Phi_x, gr_Phi_y1, gr_Phi_x_low,gr_Phi_x_high, gr_Phi_y1_low, gr_Phi_y1_high) 
    plot_gr(gr_E_mean, 'gr_E_mean', {'X':'E (GeV)','Y':'mean'})
    plot_gr(gr_E_std , 'gr_E_std' , {'X':'E (GeV)','Y':'#sigma'})
    plot_gr(gr_Theta_mean, 'gr_Theta_mean', {'X':'#theta (degree)','Y':'mean(#Delta #theta)'})
    plot_gr(gr_Theta_std , 'gr_Theta_std' , {'X':'#theta (degree)','Y':'#sigma(#Delta #theta)'})
    plot_gr(gr_Phi_mean, 'gr_Phi_mean', {'X':'#phi (degree)','Y':'mean(#Delta #phi)'})
    plot_gr(gr_Phi_std , 'gr_Phi_std' , {'X':'#phi (degree)','Y':'#sigma(#Delta #phi)'})

    print('done') 
    '''
