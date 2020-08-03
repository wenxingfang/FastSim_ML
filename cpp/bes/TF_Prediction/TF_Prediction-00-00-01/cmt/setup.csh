# echo "setup TF_Prediction TF_Prediction-00-00-01 in /junofs/users/wxfang/FastSim/bes3/workarea/BesExamples"

if ( $?CMTROOT == 0 ) then
  setenv CMTROOT /afs/ihep.ac.cn/bes3/offline/ExternalLib/SLC6/contrib/CMT/v1r25
endif
source ${CMTROOT}/mgr/setup.csh
set cmtTF_Predictiontempfile=`${CMTROOT}/mgr/cmt -quiet build temporary_name`
if $status != 0 then
  set cmtTF_Predictiontempfile=/tmp/cmt.$$
endif
${CMTROOT}/mgr/cmt setup -csh -pack=TF_Prediction -version=TF_Prediction-00-00-01 -path=/junofs/users/wxfang/FastSim/bes3/workarea/BesExamples  -no_cleanup $* >${cmtTF_Predictiontempfile}
if ( $status != 0 ) then
  echo "${CMTROOT}/mgr/cmt setup -csh -pack=TF_Prediction -version=TF_Prediction-00-00-01 -path=/junofs/users/wxfang/FastSim/bes3/workarea/BesExamples  -no_cleanup $* >${cmtTF_Predictiontempfile}"
  set cmtsetupstatus=2
  /bin/rm -f ${cmtTF_Predictiontempfile}
  unset cmtTF_Predictiontempfile
  exit $cmtsetupstatus
endif
set cmtsetupstatus=0
source ${cmtTF_Predictiontempfile}
if ( $status != 0 ) then
  set cmtsetupstatus=2
endif
/bin/rm -f ${cmtTF_Predictiontempfile}
unset cmtTF_Predictiontempfile
exit $cmtsetupstatus

