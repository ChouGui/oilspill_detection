#!/bin/bash
dl_logs(){
   #CURRENTDATE=`date +"%m-%d %T"`

   # shellcheck disable=SC2046
   scp -P 2222 -r glegat@cassiopee:/home/glegat/code/oilspill_detection/result /Users/guillaume/Desktop/MemRes/res"$1"
}
#expect "Enter passphrase for key '/Users/guillaume/.ssh/id_rsa': "
#send -- "Choupadou6"
#expect "Enter passphrase for key '/Users/guillaume/.ssh/id_rsa':"
#send -- "Choupadou6"
dl_logs "$1"
