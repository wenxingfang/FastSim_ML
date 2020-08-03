import ROOT as rt
from array import array
import gc

def do_plot(hist,out_name,title, str_para):
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
    str_bin = out_name.replace('fit_','')
    str_bin = str_bin.replace('Theta','#theta')
    str_bin = str_bin.replace('Phi','#phi')
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
    fit_range_dict['E_30_40_E'] = [-0.05, 0.05]
    fit_range_dict['E_40_50_E'] = [-0.05, 0.05]
    fit_range_dict['E_50_60_E'] = [-0.05, 0.05]
    fit_range_dict['E_60_70_E'] = [-0.05, 0.05]
    fit_range_dict['E_70_80_E'] = [-0.05, 0.05]
    fit_range_dict['E_80_90_E'] = [-0.05, 0.05]
    fit_range_dict['E_90_95_Theta'] = [-0.02, 0.02]
    fit_range_dict['E_95_100_Theta'] = [-0.02, 0.02]
    fit_range_dict['Phi_0_10_Phi'] = [-0.03, 0.03]
    fit_range_dict['Phi_-10_0_Phi'] = [-0.04, 0.04]
    fit_range_dict['Phi_10_20_Phi'] = [-0.025, 0.03]
    fit_range_dict['Phi_-20_-10_Phi'] = [-0.025, 0.03]
    fit_range_dict['Phi_20_30_Phi'] = [-0.03, 0.03]
    fit_range_dict['Phi_-30_-20_Phi'] = [-0.03, 0.03]
    fit_range_dict['Phi_30_40_Phi'] = [-0.03, 0.03]
    fit_range_dict['Phi_-40_-30_Phi'] = [-0.025, 0.03]
    fit_range_dict['Phi_50_60_Phi'] = [-0.025, 0.03]
    fit_range_dict['Phi_-60_-50_Phi'] = [-0.025, 0.03]
    fit_range_dict['Phi_60_70_Phi'] = [-0.025, 0.03]
    fit_range_dict['Phi_-70_-60_Phi'] = [-0.025, 0.03]
    fit_range_dict['Phi_70_80_Phi'] = [-0.025, 0.03]
    fit_range_dict['Phi_-80_-70_Phi'] = [-0.025, 0.03]
    fit_range_dict['Phi_-90_-80_Phi'] = [-0.03, 0.03]
    fit_range_dict['Phi_0_10_E']    = [-0.05, 0.035]
    fit_range_dict['Phi_-10_0_E']   = [-0.05, 0.035]
    fit_range_dict['Phi_10_20_E']   = [-0.05, 0.035]
    fit_range_dict['Phi_-20_-10_E'] = [-0.05, 0.035]
    fit_range_dict['Phi_20_30_E']   = [-0.05, 0.035]
    fit_range_dict['Phi_-30_-20_E'] = [-0.05, 0.035]
    fit_range_dict['Phi_30_40_E']   = [-0.05, 0.035]
    fit_range_dict['Phi_-40_-30_E'] = [-0.05, 0.035]
    fit_range_dict['Phi_40_50_E']   = [-0.05, 0.035]
    fit_range_dict['Phi_-50_-40_E'] = [-0.05, 0.035]
    fit_range_dict['Phi_50_60_E']   = [-0.05, 0.035]
    fit_range_dict['Phi_-60_-50_E'] = [-0.05, 0.035]
    fit_range_dict['Phi_60_70_E']   = [-0.05, 0.035]
    fit_range_dict['Phi_-70_-60_E'] = [-0.05, 0.035]
    fit_range_dict['Phi_70_80_E']   = [-0.05, 0.035]
    fit_range_dict['Phi_-80_-70_E'] = [-0.05, 0.035]
    fit_range_dict['Phi_80_90_E']   = [-0.05, 0.035]
    fit_range_dict['Phi_-90_-80_E'] = [-0.05, 0.035]
    fit_range_dict['Phi_0_10_Theta']    = [-0.027, 0.027]
    fit_range_dict['Phi_-10_0_Theta']   = [-0.027, 0.027]
    fit_range_dict['Phi_10_20_Theta']   = [-0.027, 0.027]
    fit_range_dict['Phi_-20_-10_Theta'] = [-0.027, 0.027]
    fit_range_dict['Phi_20_30_Theta']   = [-0.027, 0.027]
    fit_range_dict['Phi_-30_-20_Theta'] = [-0.027, 0.027]
    fit_range_dict['Phi_30_40_Theta']   = [-0.027, 0.027]
    fit_range_dict['Phi_-40_-30_Theta'] = [-0.027, 0.027]
    fit_range_dict['Phi_40_50_Theta']   = [-0.027, 0.027]
    fit_range_dict['Phi_-50_-40_Theta'] = [-0.027, 0.027]
    fit_range_dict['Phi_50_60_Theta']   = [-0.027, 0.027]
    fit_range_dict['Phi_-60_-50_Theta'] = [-0.027, 0.027]
    fit_range_dict['Phi_60_70_Theta']   = [-0.027, 0.027]
    fit_range_dict['Phi_-70_-60_Theta'] = [-0.027, 0.027]
    fit_range_dict['Phi_70_80_Theta']   = [-0.027, 0.027]
    fit_range_dict['Phi_-80_-70_Theta'] = [-0.027, 0.027]
    fit_range_dict['Phi_80_90_Theta']   = [-0.027, 0.027]
    fit_range_dict['Phi_-90_-80_Theta'] = [-0.027, 0.027]
    fit_range_dict['Theta_0_10_Theta'] = [-0.02, 0.025]
    fit_range_dict['Theta_10_20_Theta'] = [-0.02, 0.025]
    fit_range_dict['Theta_20_30_Theta'] = [-0.02, 0.02]
    fit_range_dict['Theta_30_40_Theta'] = [-0.02, 0.02]
    fit_range_dict['Theta_40_50_Theta'] = [-0.025, 0.02]
    fit_range_dict['Theta_80_90_Theta'] = [-0.03, 0.03]
    fit_range_dict['Theta_90_100_Theta'] = [-0.03, 0.03]
    fit_range_dict['Theta_120_130_Theta'] = [-0.02, 0.027]
    fit_range_dict['Theta_130_140_Theta'] = [-0.02, 0.027]
    fit_range_dict['Theta_140_150_Theta'] = [-0.02, 0.025]
    fit_range_dict['Theta_150_160_Theta'] = [-0.025, 0.025]
    fit_range_dict['Theta_160_170_Theta'] = [-0.025, 0.025]
    fit_range_dict['Theta_170_180_Theta'] = [-0.025, 0.025]
    fit_range_dict['Theta_0_10_E']    = [-0.05, 0.035]
    fit_range_dict['Theta_10_20_E']   = [-0.05, 0.035]
    fit_range_dict['Theta_20_30_E']   = [-0.05, 0.035]
    fit_range_dict['Theta_30_40_E']   = [-0.05, 0.035]
    fit_range_dict['Theta_40_50_E']   = [-0.05, 0.035]
    fit_range_dict['Theta_50_60_E']   = [-0.05, 0.035]
    fit_range_dict['Theta_60_70_E']   = [-0.05, 0.035]
    fit_range_dict['Theta_70_80_E']   = [-0.05, 0.035]
    fit_range_dict['Theta_80_90_E']   = [-0.05, 0.035]
    fit_range_dict['Theta_90_100_E']  = [-0.05, 0.035]
    fit_range_dict['Theta_100_110_E']  = [-0.05, 0.035]
    fit_range_dict['Theta_110_120_E']  = [-0.05, 0.035]
    fit_range_dict['Theta_120_130_E'] = [-0.05, 0.035]
    fit_range_dict['Theta_130_140_E'] = [-0.05, 0.035]
    fit_range_dict['Theta_140_150_E'] = [-0.05, 0.035]
    fit_range_dict['Theta_150_160_E'] = [-0.05, 0.035]
    fit_range_dict['Theta_160_170_E'] = [-0.05, 0.035]
    fit_range_dict['Theta_170_180_E'] = [-0.05, 0.035]
    
    plot_path = './plots_fit'
    #f_in= rt.TFile('full_out.root','read')
    #f_out= rt.TFile('full_res.root','RECREATE')
    f_in= rt.TFile('calo_out.root','read')
    f_out= rt.TFile('calo_res.root','RECREATE')
    para_dict = {}
    for ih in range(0,f_in.GetListOfKeys().GetSize()):
        hname=f_in.GetListOfKeys()[ih].GetName()
        if 'Theta_0_10' in hname or 'Theta_170_180' in hname:continue
        hist = f_in.Get(hname)
        mean = hist.GetMean()
        rms = hist.GetRMS()
        lower  = mean - 1.3*rms
        higher = mean + 1.3*rms
        
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
        if '_E' in hname:
            str_x = '(E_{rec} - E_{mc})/E_{mc}' 
        elif '_Theta' in hname:
            str_x = '#theta_{rec} - #theta_{mc} (degree)' 
        elif '_Phi' in hname:
            str_x = '#phi_{rec} - #phi_{mc} (degree)' 
        do_plot(hist, 'fit_%s'%hname, {'X':str_x,'Y':'Events'}, str_para)
        para_dict[hname] = [par1, err1, par2, err2]
    f_in.Close()
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
    gr_Thetabin_E_x       = array('f')
    gr_Thetabin_E_x_low   = array('f')
    gr_Thetabin_E_x_high  = array('f')
    gr_Thetabin_E_y0      = array('f')
    gr_Thetabin_E_y0_low  = array('f')
    gr_Thetabin_E_y0_high = array('f')
    gr_Thetabin_E_y1      = array('f')
    gr_Thetabin_E_y1_low  = array('f')
    gr_Thetabin_E_y1_high = array('f')
    gr_Thetabin_Theta_x       = array('f')
    gr_Thetabin_Theta_x_low   = array('f')
    gr_Thetabin_Theta_x_high  = array('f')
    gr_Thetabin_Theta_y0      = array('f')
    gr_Thetabin_Theta_y0_low  = array('f')
    gr_Thetabin_Theta_y0_high = array('f')
    gr_Thetabin_Theta_y1      = array('f')
    gr_Thetabin_Theta_y1_low  = array('f')
    gr_Thetabin_Theta_y1_high = array('f')
    gr_Thetabin_Phi_x       = array('f')
    gr_Thetabin_Phi_x_low   = array('f')
    gr_Thetabin_Phi_x_high  = array('f')
    gr_Thetabin_Phi_y0      = array('f')
    gr_Thetabin_Phi_y0_low  = array('f')
    gr_Thetabin_Phi_y0_high = array('f')
    gr_Thetabin_Phi_y1      = array('f')
    gr_Thetabin_Phi_y1_low  = array('f')
    gr_Thetabin_Phi_y1_high = array('f')
    gr_Phibin_E_x         = array('f')
    gr_Phibin_E_x_low     = array('f')
    gr_Phibin_E_x_high    = array('f')
    gr_Phibin_E_y0        = array('f')
    gr_Phibin_E_y0_low    = array('f')
    gr_Phibin_E_y0_high   = array('f')
    gr_Phibin_E_y1        = array('f')
    gr_Phibin_E_y1_low    = array('f')
    gr_Phibin_E_y1_high   = array('f')
    gr_Phibin_Theta_x         = array('f')
    gr_Phibin_Theta_x_low     = array('f')
    gr_Phibin_Theta_x_high    = array('f')
    gr_Phibin_Theta_y0        = array('f')
    gr_Phibin_Theta_y0_low    = array('f')
    gr_Phibin_Theta_y0_high   = array('f')
    gr_Phibin_Theta_y1        = array('f')
    gr_Phibin_Theta_y1_low    = array('f')
    gr_Phibin_Theta_y1_high   = array('f')
    gr_Phibin_Phi_x         = array('f')
    gr_Phibin_Phi_x_low     = array('f')
    gr_Phibin_Phi_x_high    = array('f')
    gr_Phibin_Phi_y0        = array('f')
    gr_Phibin_Phi_y0_low    = array('f')
    gr_Phibin_Phi_y0_high   = array('f')
    gr_Phibin_Phi_y1        = array('f')
    gr_Phibin_Phi_y1_low    = array('f')
    gr_Phibin_Phi_y1_high   = array('f')
    for name in para_dict:
        if 'E_' in name:
            if '_E' in name:
                low = float(name.split('_')[1])
                high = float(name.split('_')[2])
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
            elif '_Phi' in name:
                low = float(name.split('_')[1])
                high = float(name.split('_')[2])
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
            elif '_Theta' in name:
                low = float(name.split('_')[1])
                high = float(name.split('_')[2])
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
        elif 'Theta_' in name:
            if '_E' in name:
                low = float(name.split('_')[1])
                high = float(name.split('_')[2])
                mean = (low + high)/2
                gr_Thetabin_E_x.append(mean) 
                gr_Thetabin_E_x_low.append(mean-low) 
                gr_Thetabin_E_x_high.append(high-mean) 
                gr_Thetabin_E_y0.append(para_dict[name][0]) 
                gr_Thetabin_E_y0_low.append(para_dict[name][1]) 
                gr_Thetabin_E_y0_high.append(para_dict[name][1]) 
                gr_Thetabin_E_y1.append(para_dict[name][2]) 
                gr_Thetabin_E_y1_low.append(para_dict[name] [3]) 
                gr_Thetabin_E_y1_high.append(para_dict[name][3]) 
            elif '_Phi' in name:
                low = float(name.split('_')[1])
                high = float(name.split('_')[2])
                mean = (low + high)/2
                gr_Thetabin_Phi_x.append(mean) 
                gr_Thetabin_Phi_x_low.append(mean-low) 
                gr_Thetabin_Phi_x_high.append(high-mean) 
                gr_Thetabin_Phi_y0.append(para_dict[name][0]) 
                gr_Thetabin_Phi_y0_low.append(para_dict[name][1]) 
                gr_Thetabin_Phi_y0_high.append(para_dict[name][1]) 
                gr_Thetabin_Phi_y1.append(para_dict[name][2]) 
                gr_Thetabin_Phi_y1_low.append(para_dict[name] [3]) 
                gr_Thetabin_Phi_y1_high.append(para_dict[name][3]) 
            elif '_Theta' in name:
                low = float(name.split('_')[1])
                high = float(name.split('_')[2])
                mean = (low + high)/2
                gr_Thetabin_Theta_x.append(mean) 
                gr_Thetabin_Theta_x_low.append(mean-low) 
                gr_Thetabin_Theta_x_high.append(high-mean) 
                gr_Thetabin_Theta_y0.append(para_dict[name][0]) 
                gr_Thetabin_Theta_y0_low.append(para_dict[name][1]) 
                gr_Thetabin_Theta_y0_high.append(para_dict[name][1]) 
                gr_Thetabin_Theta_y1.append(para_dict[name][2]) 
                gr_Thetabin_Theta_y1_low.append(para_dict[name] [3]) 
                gr_Thetabin_Theta_y1_high.append(para_dict[name][3]) 
        elif 'Phi_' in name:
            if '_E' in name:
                low = float(name.split('_')[1])
                high = float(name.split('_')[2])
                mean = (low + high)/2
                gr_Phibin_E_x.append(mean) 
                gr_Phibin_E_x_low.append(mean-low) 
                gr_Phibin_E_x_high.append(high-mean) 
                gr_Phibin_E_y0.append(para_dict[name][0]) 
                gr_Phibin_E_y0_low.append(para_dict[name][1]) 
                gr_Phibin_E_y0_high.append(para_dict[name][1]) 
                gr_Phibin_E_y1.append(para_dict[name][2]) 
                gr_Phibin_E_y1_low.append(para_dict[name] [3]) 
                gr_Phibin_E_y1_high.append(para_dict[name][3]) 
            elif '_Phi' in name:
                low = float(name.split('_')[1])
                high = float(name.split('_')[2])
                mean = (low + high)/2
                gr_Phibin_Phi_x.append(mean) 
                gr_Phibin_Phi_x_low.append(mean-low) 
                gr_Phibin_Phi_x_high.append(high-mean) 
                gr_Phibin_Phi_y0.append(para_dict[name][0]) 
                gr_Phibin_Phi_y0_low.append(para_dict[name][1]) 
                gr_Phibin_Phi_y0_high.append(para_dict[name][1]) 
                gr_Phibin_Phi_y1.append(para_dict[name][2]) 
                gr_Phibin_Phi_y1_low.append(para_dict[name] [3]) 
                gr_Phibin_Phi_y1_high.append(para_dict[name][3]) 
            elif '_Theta' in name:
                low = float(name.split('_')[1])
                high = float(name.split('_')[2])
                mean = (low + high)/2
                gr_Phibin_Theta_x.append(mean) 
                gr_Phibin_Theta_x_low.append(mean-low) 
                gr_Phibin_Theta_x_high.append(high-mean) 
                gr_Phibin_Theta_y0.append(para_dict[name][0]) 
                gr_Phibin_Theta_y0_low.append(para_dict[name][1]) 
                gr_Phibin_Theta_y0_high.append(para_dict[name][1]) 
                gr_Phibin_Theta_y1.append(para_dict[name][2]) 
                gr_Phibin_Theta_y1_low.append(para_dict[name] [3]) 
                gr_Phibin_Theta_y1_high.append(para_dict[name][3]) 
           
    gr_Ebin_E_mean     = rt.TGraphAsymmErrors(int(len(gr_Ebin_E_x)),gr_Ebin_E_x, gr_Ebin_E_y0, gr_Ebin_E_x_low,gr_Ebin_E_x_high, gr_Ebin_E_y0_low, gr_Ebin_E_y0_high) 
    gr_Ebin_E_std      = rt.TGraphAsymmErrors(int(len(gr_Ebin_E_x)),gr_Ebin_E_x, gr_Ebin_E_y1, gr_Ebin_E_x_low,gr_Ebin_E_x_high, gr_Ebin_E_y1_low, gr_Ebin_E_y1_high) 
    gr_Ebin_Theta_mean = rt.TGraphAsymmErrors(int(len(gr_Ebin_Theta_x)),gr_Ebin_Theta_x, gr_Ebin_Theta_y0, gr_Ebin_Theta_x_low,gr_Ebin_Theta_x_high, gr_Ebin_Theta_y0_low, gr_Ebin_Theta_y0_high) 
    gr_Ebin_Theta_std  = rt.TGraphAsymmErrors(int(len(gr_Ebin_Theta_x)),gr_Ebin_Theta_x, gr_Ebin_Theta_y1, gr_Ebin_Theta_x_low,gr_Ebin_Theta_x_high, gr_Ebin_Theta_y1_low, gr_Ebin_Theta_y1_high) 
    gr_Ebin_Phi_mean = rt.TGraphAsymmErrors(int(len(gr_Ebin_Phi_x)),gr_Ebin_Phi_x, gr_Ebin_Phi_y0, gr_Ebin_Phi_x_low,gr_Ebin_Phi_x_high, gr_Ebin_Phi_y0_low, gr_Ebin_Phi_y0_high) 
    gr_Ebin_Phi_std  = rt.TGraphAsymmErrors(int(len(gr_Ebin_Phi_x)),gr_Ebin_Phi_x, gr_Ebin_Phi_y1, gr_Ebin_Phi_x_low,gr_Ebin_Phi_x_high, gr_Ebin_Phi_y1_low, gr_Ebin_Phi_y1_high) 
    gr_Thetabin_E_mean     = rt.TGraphAsymmErrors(int(len(gr_Thetabin_E_x)),gr_Thetabin_E_x, gr_Thetabin_E_y0, gr_Thetabin_E_x_low,gr_Thetabin_E_x_high, gr_Thetabin_E_y0_low, gr_Thetabin_E_y0_high) 
    gr_Thetabin_E_std      = rt.TGraphAsymmErrors(int(len(gr_Thetabin_E_x)),gr_Thetabin_E_x, gr_Thetabin_E_y1, gr_Thetabin_E_x_low,gr_Thetabin_E_x_high, gr_Thetabin_E_y1_low, gr_Thetabin_E_y1_high) 
    gr_Thetabin_Theta_mean = rt.TGraphAsymmErrors(int(len(gr_Thetabin_Theta_x)),gr_Thetabin_Theta_x, gr_Thetabin_Theta_y0, gr_Thetabin_Theta_x_low,gr_Thetabin_Theta_x_high, gr_Thetabin_Theta_y0_low, gr_Thetabin_Theta_y0_high) 
    gr_Thetabin_Theta_std  = rt.TGraphAsymmErrors(int(len(gr_Thetabin_Theta_x)),gr_Thetabin_Theta_x, gr_Thetabin_Theta_y1, gr_Thetabin_Theta_x_low,gr_Thetabin_Theta_x_high, gr_Thetabin_Theta_y1_low, gr_Thetabin_Theta_y1_high) 
    gr_Thetabin_Phi_mean = rt.TGraphAsymmErrors(int(len(gr_Thetabin_Phi_x)),gr_Thetabin_Phi_x, gr_Thetabin_Phi_y0, gr_Thetabin_Phi_x_low,gr_Thetabin_Phi_x_high, gr_Thetabin_Phi_y0_low, gr_Thetabin_Phi_y0_high) 
    gr_Thetabin_Phi_std  = rt.TGraphAsymmErrors(int(len(gr_Thetabin_Phi_x)),gr_Thetabin_Phi_x, gr_Thetabin_Phi_y1, gr_Thetabin_Phi_x_low,gr_Thetabin_Phi_x_high, gr_Thetabin_Phi_y1_low, gr_Thetabin_Phi_y1_high) 
    gr_Phibin_E_mean     = rt.TGraphAsymmErrors(int(len(gr_Phibin_E_x)),gr_Phibin_E_x, gr_Phibin_E_y0, gr_Phibin_E_x_low,gr_Phibin_E_x_high, gr_Phibin_E_y0_low, gr_Phibin_E_y0_high) 
    gr_Phibin_E_std      = rt.TGraphAsymmErrors(int(len(gr_Phibin_E_x)),gr_Phibin_E_x, gr_Phibin_E_y1, gr_Phibin_E_x_low,gr_Phibin_E_x_high, gr_Phibin_E_y1_low, gr_Phibin_E_y1_high) 
    gr_Phibin_Theta_mean = rt.TGraphAsymmErrors(int(len(gr_Phibin_Theta_x)),gr_Phibin_Theta_x, gr_Phibin_Theta_y0, gr_Phibin_Theta_x_low,gr_Phibin_Theta_x_high, gr_Phibin_Theta_y0_low, gr_Phibin_Theta_y0_high) 
    gr_Phibin_Theta_std  = rt.TGraphAsymmErrors(int(len(gr_Phibin_Theta_x)),gr_Phibin_Theta_x, gr_Phibin_Theta_y1, gr_Phibin_Theta_x_low,gr_Phibin_Theta_x_high, gr_Phibin_Theta_y1_low, gr_Phibin_Theta_y1_high) 
    gr_Phibin_Phi_mean = rt.TGraphAsymmErrors(int(len(gr_Phibin_Phi_x)),gr_Phibin_Phi_x, gr_Phibin_Phi_y0, gr_Phibin_Phi_x_low,gr_Phibin_Phi_x_high, gr_Phibin_Phi_y0_low, gr_Phibin_Phi_y0_high) 
    gr_Phibin_Phi_std  = rt.TGraphAsymmErrors(int(len(gr_Phibin_Phi_x)),gr_Phibin_Phi_x, gr_Phibin_Phi_y1, gr_Phibin_Phi_x_low,gr_Phibin_Phi_x_high, gr_Phibin_Phi_y1_low, gr_Phibin_Phi_y1_high) 
    plot_gr(gr_Ebin_E_mean    , 'gr_Ebin_E_mean'    , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'mean(#DeltaE/E)'})
    plot_gr(gr_Ebin_E_std     , 'gr_Ebin_E_std'     , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'#sigma(#DeltaE/E)'})
    plot_gr(gr_Thetabin_E_mean    , 'gr_Thetabin_E_mean'    , {'X':'#theta_{pfo}^{#gamma} (degree)','Y':'mean(#DeltaE/E)'})
    plot_gr(gr_Thetabin_E_std     , 'gr_Thetabin_E_std'     , {'X':'#theta_{pfo}^{#gamma} (degree)','Y':'#sigma(#DeltaE/E)'})
    plot_gr(gr_Phibin_E_mean    , 'gr_Phibin_E_mean'    , {'X':'#phi_{pfo}^{#gamma} (degree)','Y':'mean(#DeltaE/E)'})
    plot_gr(gr_Phibin_E_std     , 'gr_Phibin_E_std'     , {'X':'#phi_{pfo}^{#gamma} (degree)','Y':'#sigma(#DeltaE/E)'})
    plot_gr(gr_Ebin_Theta_mean, 'gr_Ebin_Theta_mean', {'X':'E_{pfo}^{#gamma} (GeV)','Y':'mean(#Delta #theta)'})
    plot_gr(gr_Ebin_Theta_std , 'gr_Ebin_Theta_std' , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'#sigma(#Delta #theta)'})
    plot_gr(gr_Thetabin_Theta_mean, 'gr_Thetabin_Theta_mean', {'X':'#theta_{pfo}^{#gamma} (degree)','Y':'mean(#Delta #theta)'})
    plot_gr(gr_Thetabin_Theta_std , 'gr_Thetabin_Theta_std' , {'X':'#theta_{pfo}^{#gamma} (degree)','Y':'#sigma(#Delta #theta)'})
    plot_gr(gr_Phibin_Theta_mean, 'gr_Phibin_Theta_mean', {'X':'#phi_{pfo}^{#gamma} (degree)','Y':'mean(#Delta #theta)'})
    plot_gr(gr_Phibin_Theta_std , 'gr_Phibin_Theta_std' , {'X':'#phi_{pfo}^{#gamma} (degree)','Y':'#sigma(#Delta #theta)'})
    plot_gr(gr_Ebin_Phi_mean  , 'gr_Ebin_Phi_mean'  , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'mean(#Delta #phi)'})
    plot_gr(gr_Ebin_Phi_std   , 'gr_Ebin_Phi_std'   , {'X':'E_{pfo}^{#gamma} (GeV)','Y':'#sigma(#Delta #phi)'})
    plot_gr(gr_Thetabin_Phi_mean  , 'gr_Thetabin_Phi_mean'  , {'X':'#theta_{pfo}^{#gamma} (degree)','Y':'mean(#Delta #phi)'})
    plot_gr(gr_Thetabin_Phi_std   , 'gr_Thetabin_Phi_std'   , {'X':'#theta_{pfo}^{#gamma} (degree)','Y':'#sigma(#Delta #phi)'})
    plot_gr(gr_Phibin_Phi_mean  , 'gr_Phibin_Phi_mean'  , {'X':'#phi_{pfo}^{#gamma} (degree)','Y':'mean(#Delta #phi)'})
    plot_gr(gr_Phibin_Phi_std   , 'gr_Phibin_Phi_std'   , {'X':'#phi_{pfo}^{#gamma} (degree)','Y':'#sigma(#Delta #phi)'})
    f_out.cd()
    gr_Ebin_E_mean     .Write('gr_Ebin_E_mean'    )
    gr_Ebin_E_std      .Write('gr_Ebin_E_std'     )
    gr_Ebin_Theta_mean .Write('gr_Ebin_Theta_mean')
    gr_Ebin_Theta_std  .Write('gr_Ebin_Theta_std' )
    gr_Ebin_Phi_mean   .Write('gr_Ebin_Phi_mean'  )
    gr_Ebin_Phi_std    .Write('gr_Ebin_Phi_std'   )
    gr_Thetabin_E_mean     .Write('gr_Thetabin_E_mean'    )
    gr_Thetabin_E_std      .Write('gr_Thetabin_E_std'     )
    gr_Thetabin_Theta_mean .Write('gr_Thetabin_Theta_mean')
    gr_Thetabin_Theta_std  .Write('gr_Thetabin_Theta_std' )
    gr_Thetabin_Phi_mean   .Write('gr_Thetabin_Phi_mean'  )
    gr_Thetabin_Phi_std    .Write('gr_Thetabin_Phi_std'   )
    gr_Phibin_E_mean     .Write('gr_Phibin_E_mean'    )
    gr_Phibin_E_std      .Write('gr_Phibin_E_std'     )
    gr_Phibin_Theta_mean .Write('gr_Phibin_Theta_mean')
    gr_Phibin_Theta_std  .Write('gr_Phibin_Theta_std' )
    gr_Phibin_Phi_mean   .Write('gr_Phibin_Phi_mean'  )
    gr_Phibin_Phi_std    .Write('gr_Phibin_Phi_std'   )
    f_out.Close() 
    print('done') 
