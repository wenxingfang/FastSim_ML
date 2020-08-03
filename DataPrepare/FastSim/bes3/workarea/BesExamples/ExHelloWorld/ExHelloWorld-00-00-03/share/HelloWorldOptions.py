###############################################################
#
# Job options file
#
#==============================================================

#--------------------------------------------------------------
# ATLAS default Application Configuration options
#--------------------------------------------------------------

theApp.setup( MONTECARLO )

#--------------------------------------------------------------
# Private Application Configuration options
#--------------------------------------------------------------

#load relevant libraries
theApp.Dlls += [ "ExHelloWorld" ]

#top algorithms to be run
theApp.TopAlg = [ "HelloWorld" ]
HelloWorld = Algorithm( "HelloWorld" )

#--------------------------------------------------------------
# Set output level threshold (DEBUG, INFO, WARNING, ERROR, FATAL)
#--------------------------------------------------------------

MessageSvc.OutputLevel = INFO

#--------------------------------------------------------------
# Event related parameters
#--------------------------------------------------------------

# Number of events to be processed (default is 10)
theApp.EvtMax = 10

#--------------------------------------------------------------
# Algorithms Private Options
#--------------------------------------------------------------

# For the HelloWorld algorithm
HelloWorld.MyInt = 42
HelloWorld.MyBool = 1
HelloWorld.MyDouble = 3.14159
HelloWorld.MyStringVec = [ "Welcome", "to", "Athena", "Framework", "Tutorial" ]

#--------------------------------------------------------------
# Batch/Interactive Control (uncomment the lines for batch mode)
#--------------------------------------------------------------

####theApp.run( theApp.EvtMax )
####theApp.exit()

#==============================================================
#
# End of job options file
#
###############################################################


