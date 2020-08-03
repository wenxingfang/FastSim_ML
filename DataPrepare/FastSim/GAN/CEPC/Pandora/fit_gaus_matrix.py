import ROOT as rt
from array import array
import gc

def do_plot(hist,out_name,title, str_para, str_bin):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.15)
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
    #hist.GetYaxis().SetTitle(str(title['Y']+' / %.1f %s'%(hist.GetBinWidth(1), ystr)))
    hist.GetYaxis().SetTitle(str(title['Y']))
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.SetLineColor(rt.kBlue)
    hist.SetMarkerColor(rt.kBlue)
    hist.SetMarkerStyle(20)
    hist.Draw("histep")
    func = hist.GetFunction("f1")
    func.Draw('SAME')
    label_f=rt.TLatex(0.15, 0.89 , "%s"%(str(str_para)))
    label_f.SetTextSize(0.03)
    label_f.SetNDC()
    label_f.Draw()
    #str_bin = out_name.replace('fit_','')
    #str_bin = str_bin.replace('Theta','#theta')
    #str_bin = str_bin.replace('Phi','#phi')
    label_bin=rt.TLatex(0.18, 0.8 , "%s"%(str(str_bin)))
    label_bin.SetTextSize(0.03)
    label_bin.SetNDC()
    label_bin.Draw()
    canvas.SaveAs("%s/%s.png"%(plot_path,out_name))
    del canvas
    gc.collect()

def plot_gr(hist,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    hist.SetTitle('')
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
    fit_range_dict = {}
    
    plot_path = './plots_fit'
    f_in= rt.TFile('MatrixGamma_out.root','UPDATE')
    para_dict = {}
    for ih in range(0,f_in.GetListOfKeys().GetSize()):
        hname=f_in.GetListOfKeys()[ih].GetName()
        if '_res_' not in hname :continue
        energy = hname.split('_')[-1]
        energy = energy.replace('GeV','')
         
        hist = f_in.Get(hname)
        mean = hist.GetMean()
        rms = hist.GetRMS()
        lower  = mean - 2*rms
        higher = mean + 2*rms
        if'_E_res_' in hname:
            #lower  = 1.02*mean - 1.4*rms
            #higher = 1.02*mean + 1.4*rms
            lower  = 1.01*mean - 1.0*rms
            higher = 1.01*mean + 1.3*rms
        
        if hname in fit_range_dict:
            lower  = fit_range_dict[hname][0]
            higher = fit_range_dict[hname][1]
        #f1 = rt.TF1("f1", "gaus", -0.05+mean, 0.05+mean)
        f1 = rt.TF1("f1", "gaus", lower, higher)
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
        str_para = 'mean=%.3f#pm%.3f, #sigma=%.3f#pm%.3f'%(par1, err1, par2, err2)
        str_x = ''
        if '_E_' in hname:
            str_x = '(E_{rec} - E_{mc})/E_{mc}' 
        elif '_theta_' in hname:
            str_x = '#theta_{rec} - #theta_{mc} (degree)' 
        elif '_phi_' in hname:
            str_x = '#phi_{rec} - #phi_{mc} (degree)' 
        do_plot(hist, 'fit_%s'%hname, {'X':str_x,'Y':'Events'}, str_para, "E_%sGeV"%energy )
        para_dict[hname] = [par1, err1, par2, err2]
    gr_Ebin_E_x           = array('f')
    gr_Ebin_E_x_low       = array('f')
    gr_Ebin_E_x_high      = array('f')
    gr_Ebin_E_y0          = array('f')
    gr_Ebin_E_y0_low      = array('f')
    gr_Ebin_E_y0_high     = array('f')
    gr_Ebin_E_y1          = array('f')
    gr_Ebin_E_y1_low      = array('f')
    gr_Ebin_E_y1_high     = array('f')
    gr_Ebin_Theta_x       = array('f')
    gr_Ebin_Theta_x_low   = array('f')
    gr_Ebin_Theta_x_high  = array('f')
    gr_Ebin_Theta_y0      = array('f')
    gr_Ebin_Theta_y0_low  = array('f')
    gr_Ebin_Theta_y0_high = array('f')
    gr_Ebin_Theta_y1      = array('f')
    gr_Ebin_Theta_y1_low  = array('f')
    gr_Ebin_Theta_y1_high = array('f')
    gr_Ebin_Phi_x         = array('f')
    gr_Ebin_Phi_x_low     = array('f')
    gr_Ebin_Phi_x_high    = array('f')
    gr_Ebin_Phi_y0        = array('f')
    gr_Ebin_Phi_y0_low    = array('f')
    gr_Ebin_Phi_y0_high   = array('f')
    gr_Ebin_Phi_y1        = array('f')
    gr_Ebin_Phi_y1_low    = array('f')
    gr_Ebin_Phi_y1_high   = array('f')
    for name in para_dict:
        energy = name.split('_')[-1]
        energy = energy.replace('GeV','')
        energy = float(energy)
        if '_E_' in name:
            low  = energy-0.5
            high = energy+0.5
            mean = (low + high)/2
            gr_Ebin_E_x.append(mean) 
            gr_Ebin_E_x_low.append(mean-low) 
            gr_Ebin_E_x_high.append(high-mean) 
            gr_Ebin_E_y0.append(para_dict[name][0]) 
            gr_Ebin_E_y0_low.append(para_dict[name][1]) 
            gr_Ebin_E_y0_high.append(para_dict[name][1]) 
            gr_Ebin_E_y1.append(para_dict[name][2]) 
            gr_Ebin_E_y1_low.append(para_dict[name] [3]) 
            gr_Ebin_E_y1_high.append(para_dict[name][3]) 
        elif '_phi_' in name:
            low  = energy-0.5
            high = energy+0.5
            mean = (low + high)/2
            gr_Ebin_Phi_x.append(mean) 
            gr_Ebin_Phi_x_low.append(mean-low) 
            gr_Ebin_Phi_x_high.append(high-mean) 
            gr_Ebin_Phi_y0.append(para_dict[name][0]) 
            gr_Ebin_Phi_y0_low.append(para_dict[name][1]) 
            gr_Ebin_Phi_y0_high.append(para_dict[name][1]) 
            gr_Ebin_Phi_y1.append(para_dict[name][2]) 
            gr_Ebin_Phi_y1_low.append(para_dict[name] [3]) 
            gr_Ebin_Phi_y1_high.append(para_dict[name][3]) 
        elif '_theta_' in name:
            low  = energy-0.5
            high = energy+0.5
            mean = (low + high)/2
            gr_Ebin_Theta_x.append(mean) 
            gr_Ebin_Theta_x_low.append(mean-low) 
            gr_Ebin_Theta_x_high.append(high-mean) 
            gr_Ebin_Theta_y0.append(para_dict[name][0]) 
            gr_Ebin_Theta_y0_low.append(para_dict[name][1]) 
            gr_Ebin_Theta_y0_high.append(para_dict[name][1]) 
            gr_Ebin_Theta_y1.append(para_dict[name][2]) 
            gr_Ebin_Theta_y1_low.append(para_dict[name] [3]) 
            gr_Ebin_Theta_y1_high.append(para_dict[name][3]) 
           
    gr_Ebin_E_mean     = rt.TGraphAsymmErrors(int(len(gr_Ebin_E_x)),gr_Ebin_E_x, gr_Ebin_E_y0, gr_Ebin_E_x_low,gr_Ebin_E_x_high, gr_Ebin_E_y0_low, gr_Ebin_E_y0_high) 
    gr_Ebin_E_std      = rt.TGraphAsymmErrors(int(len(gr_Ebin_E_x)),gr_Ebin_E_x, gr_Ebin_E_y1, gr_Ebin_E_x_low,gr_Ebin_E_x_high, gr_Ebin_E_y1_low, gr_Ebin_E_y1_high) 
    gr_Ebin_Theta_mean = rt.TGraphAsymmErrors(int(len(gr_Ebin_Theta_x)),gr_Ebin_Theta_x, gr_Ebin_Theta_y0, gr_Ebin_Theta_x_low,gr_Ebin_Theta_x_high, gr_Ebin_Theta_y0_low, gr_Ebin_Theta_y0_high) 
    gr_Ebin_Theta_std  = rt.TGraphAsymmErrors(int(len(gr_Ebin_Theta_x)),gr_Ebin_Theta_x, gr_Ebin_Theta_y1, gr_Ebin_Theta_x_low,gr_Ebin_Theta_x_high, gr_Ebin_Theta_y1_low, gr_Ebin_Theta_y1_high) 
    gr_Ebin_Phi_mean = rt.TGraphAsymmErrors(int(len(gr_Ebin_Phi_x)),gr_Ebin_Phi_x, gr_Ebin_Phi_y0, gr_Ebin_Phi_x_low,gr_Ebin_Phi_x_high, gr_Ebin_Phi_y0_low, gr_Ebin_Phi_y0_high) 
    gr_Ebin_Phi_std  = rt.TGraphAsymmErrors(int(len(gr_Ebin_Phi_x)),gr_Ebin_Phi_x, gr_Ebin_Phi_y1, gr_Ebin_Phi_x_low,gr_Ebin_Phi_x_high, gr_Ebin_Phi_y1_low, gr_Ebin_Phi_y1_high) 
    plot_gr(gr_Ebin_E_mean    , 'gr_Ebin_E_mean'    , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'mean(#DeltaE/E)'})
    plot_gr(gr_Ebin_E_std     , 'gr_Ebin_E_std'     , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'#sigma(#DeltaE/E)'})
    plot_gr(gr_Ebin_Theta_mean, 'gr_Ebin_Theta_mean', {'X':'E_{pfo}^{#gamma} (GeV)','Y':'mean(#Delta #theta)'})
    plot_gr(gr_Ebin_Theta_std , 'gr_Ebin_Theta_std' , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'#sigma(#Delta #theta)'})
    plot_gr(gr_Ebin_Phi_mean  , 'gr_Ebin_Phi_mean'  , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'mean(#Delta #phi)'})
    plot_gr(gr_Ebin_Phi_std   , 'gr_Ebin_Phi_std'   , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'#sigma(#Delta #phi)'})
    f_in.cd()
    gr_Ebin_E_mean     .Write('gr_Ebin_E_mean'    )
    gr_Ebin_E_std      .Write('gr_Ebin_E_std'     )
    gr_Ebin_Theta_mean .Write('gr_Ebin_Theta_mean')
    gr_Ebin_Theta_std  .Write('gr_Ebin_Theta_std' )
    gr_Ebin_Phi_mean   .Write('gr_Ebin_Phi_mean'  )
    gr_Ebin_Phi_std    .Write('gr_Ebin_Phi_std'   )
    f_in.Close() 
    print('done') 
