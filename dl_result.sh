#!/bin/bash
dl_logs(){
   yes Choupadou6 | scp -P 2222 -r glegat@cassiopee:/linux/glegat/code/oilspill_detection/result/* Desktop/MemRes
}
#expect "Enter passphrase for key '/Users/guillaume/.ssh/id_rsa': "
#send -- "Choupadou6"
#expect "Enter passphrase for key '/Users/guillaume/.ssh/id_rsa':"
#send -- "Choupadou6"
dl_logs
