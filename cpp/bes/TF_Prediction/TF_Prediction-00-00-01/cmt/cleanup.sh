# echo "cleanup TF_Prediction TF_Prediction-00-00-01 in /junofs/users/wxfang/FastSim/bes3/workarea/BesExamples"

if test "${CMTROOT}" = ""; then
  CMTROOT=/afs/ihep.ac.cn/bes3/offline/ExternalLib/SLC6/contrib/CMT/v1r25; export CMTROOT
fi
. ${CMTROOT}/mgr/setup.sh
cmtTF_Predictiontempfile=`${CMTROOT}/mgr/cmt -quiet build temporary_name`
if test ! $? = 0 ; then cmtTF_Predictiontempfile=/tmp/cmt.$$; fi
${CMTROOT}/mgr/cmt cleanup -sh -pack=TF_Prediction -version=TF_Prediction-00-00-01 -path=/junofs/users/wxfang/FastSim/bes3/workarea/BesExamples  $* >${cmtTF_Predictiontempfile}
if test $? != 0 ; then
  echo >&2 "${CMTROOT}/mgr/cmt cleanup -sh -pack=TF_Prediction -version=TF_Prediction-00-00-01 -path=/junofs/users/wxfang/FastSim/bes3/workarea/BesExamples  $* >${cmtTF_Predictiontempfile}"
  cmtcleanupstatus=2
  /bin/rm -f ${cmtTF_Predictiontempfile}
  unset cmtTF_Predictiontempfile
  return $cmtcleanupstatus
fi
cmtcleanupstatus=0
. ${cmtTF_Predictiontempfile}
if test $? != 0 ; then
  cmtcleanupstatus=2
fi
/bin/rm -f ${cmtTF_Predictiontempfile}
unset cmtTF_Predictiontempfile
return $cmtcleanupstatus

