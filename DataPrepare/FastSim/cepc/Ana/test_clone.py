import ROOT as rt

if False:

    f_out = rt.TFile('test_clone.root','RECREATE')
    
    h = rt.TH1F('test','',20,0,20)
    
    for i in range(10):
        h.SetBinContent(i+1, 2)
    
    f_out.cd()
    h.Write()
    f_out.Close()

if True:

    f_in = rt.TFile('test_clone.root','READ')
    h = f_in.Get('test')
    h1 = h.Clone('h1')
    h2 = h.Clone('h2')
    h3 = h.Clone('h3')
    for i in range(1,11):
        print('before change, bin=',i,',h=',h.GetBinContent(i),',h1=',h1.GetBinContent(i),',h2=',h2.GetBinContent(i),',h3=',h3.GetBinContent(i))
    for i in range(1,11):
        h.SetBinContent(i, 3)
    for i in range(1,11):
        print('after change h to 3, bin=',i,',h=',h.GetBinContent(i),',h1=',h1.GetBinContent(i),',h2=',h2.GetBinContent(i),',h3=',h3.GetBinContent(i))
    for i in range(1,11):
        h1.SetBinContent(i, 10)
    for i in range(1,11):
        print('after change h1 to 10, bin=',i,',h=',h.GetBinContent(i),',h1=',h1.GetBinContent(i),',h2=',h2.GetBinContent(i),',h3=',h3.GetBinContent(i))
    dicts = {}
    dicts['h'] = h
    dicts['h1'] = h1
    dicts['h2'] = h2
    dicts['h3'] = h3
    for i in range(1,11):
        print('from dicts, bin=',i,',h=',dicts['h'].GetBinContent(i),',h1=',dicts['h1'].GetBinContent(i),',h2=',dicts['h2'].GetBinContent(i),',h3=',dicts['h3'].GetBinContent(i))


