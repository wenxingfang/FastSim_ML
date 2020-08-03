# echo "cleanup EmcRec EmcRec-01-02-57 in /junofs/users/wxfang/FastSim/bes3/workarea/Reconstruction"

if ( $?CMTROOT == 0 ) then
  setenv CMTROOT /afs/ihep.ac.cn/bes3/offline/ExternalLib/SLC6/contrib/CMT/v1r25
endif
source ${CMTROOT}/mgr/setup.csh
set cmtEmcRectempfile=`${CMTROOT}/mgr/cmt -quiet build temporary_name`
if $status != 0 then
  set cmtEmcRectempfile=/tmp/cmt.$$
endif
${CMTROOT}/mgr/cmt cleanup -csh -pack=EmcRec -version=EmcRec-01-02-57 -path=/junofs/users/wxfang/FastSim/bes3/workarea/Reconstruction  $* >${cmtEmcRectempfile}
if ( $status != 0 ) then
  echo "${CMTROOT}/mgr/cmt cleanup -csh -pack=EmcRec -version=EmcRec-01-02-57 -path=/junofs/users/wxfang/FastSim/bes3/workarea/Reconstruction  $* >${cmtEmcRectempfile}"
  set cmtcleanupstatus=2
  /bin/rm -f ${cmtEmcRectempfile}
  unset cmtEmcRectempfile
  exit $cmtcleanupstatus
endif
set cmtcleanupstatus=0
source ${cmtEmcRectempfile}
if ( $status != 0 ) then
  set cmtcleanupstatus=2
endif
/bin/rm -f ${cmtEmcRectempfile}
unset cmtEmcRectempfile
exit $cmtcleanupstatus

