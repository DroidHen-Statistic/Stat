BEGIN{
    months["Jan"] = "01"
    months["Feb"] = "02"
    months["Mar"] = "03"
    months["Apr"] = "04"
    months["May"] = "05"
    months["Jun"] = "06"
    months["Jul"] = "07"
    months["Aug"] = "08"
    months["Sep"] = "09"
    months["Oct"] = "10"
    months["Nov"] = "11"
    months["Dec"] = "12"
}
function str_to_datetime(year, month, day, time){
    month = months[month]
    split(time, t, ":")
    if(day < 10){
        day = "0"day
    }
    return year""month""day""t[1]""t[2]""t[3]
}
{
#   log_format:
#
#   login datetime uid
#   spin  datetime uid is_free machine_id pay_in win coin win_bonus win_free_spin
#   purchase datetime uid
#   play_bonus datetime uid machine_id total_win coin


    if($0 ~ /u'cmd':\s*u'login'/){ 
       # printf "login "
        printf "1 "
        printf str_to_datetime(2017,$1,$2,$3)" "
        if(match($0, /"uid":\s*[0-9]+/) > 0){
            str = substr($0,RSTART+7,RLENGTH-7)
            printf str
        }
        printf "\n"
    }

    if($0 ~ /u'cmd':\s*u'spin'/){
        #printf "spin " 
        printf "2 "
        printf str_to_datetime(2017,$1,$2,$3)" "
        if(match($0, /u'uid':\s*[0-9]+/)>0){
            str = substr($0,RSTART+8,RLENGTH-8)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /u'is_free':\s*[0-1]+/)>0){
            str = substr($0,RSTART+12,RLENGTH-12)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /u'machine_id':\s*[0-9]+/)>0){
            str = substr($0,RSTART+15,RLENGTH-15)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /"pay_in":\s*[0-9]+/)>0){
            str = substr($0,RSTART+10,RLENGTH-10)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /"win":\s*[0-9]+/)>0){
            str = substr($0,RSTART+7,RLENGTH-7)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /"coin":\s*[0-9]+/)>0){
            str = substr($0,RSTART+8,RLENGTH-8)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /"win_bonus":/)>0){
            printf "1 "
        }
        else{
            printf "0 "
        }
        if(match($0, /"win_free_spin":/)>0){
            remain = substr($0,RSTART+17)
            match(remain, /"num":\s[0-9]+/)
            str = substr(remain, RSTART+7, RLENGTH-7)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /u'bet':\s*[0-9]+/)>0){
            str = substr($0,RSTART+8,RLENGTH-8)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /u'lines':\s*[0-9]+/)>0){
            str = substr($0,RSTART+10,RLENGTH-10)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /"level":\s*[0-9]+/)>0){
            str = substr($0,RSTART+9,RLENGTH-9)
            printf str" "
        }
        else{
            printf "0 "
        }
        printf "\n"
    }

    if($0 ~ /u'cmd':\s*u'verify_purchase'/){
       # printf "purchase "
        printf "4 "
        printf str_to_datetime(2017,$1,$2,$3)" "
        if(match($0, /u'uid':\s*[0-9]+/)>0){
            str = substr($0,RSTART+8,RLENGTH-8)
            printf str" "
        }
        else{
            printf "0 "
        }
        printf "\n"    
    }

    if($0 ~ /u'cmd':\s*u'play_bonus'/){
        #printf "play_bonus "
        printf "3 "
        printf str_to_datetime(2017,$1,$2,$3)" "
        if(match($0, /u'uid':\s*[0-9]+/)>0){
            str = substr($0,RSTART+8,RLENGTH-8)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /u'machine_id':\s*[0-9]+/)>0){
            str = substr($0,RSTART+15,RLENGTH-15)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /"total_win":\s*[0-9]+/)>0){
            str = substr($0,RSTART+13,RLENGTH-13)
            printf str" "
        }
        else{
            printf "0 "
        }
        if(match($0, /"coin":\s*[0-9]+/)>0){
            str = substr($0,RSTART+8,RLENGTH-8)
            printf str" "
        }
        else{
            printf "0 "
        }
        printf "\n"
    } 
}
