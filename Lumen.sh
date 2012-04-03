#!/bin/bash

echo "Password is lumen :)"
sudo echo "LUMEN!"

if [ "$1" == "build" ]
then
	cd src
	make clean
	cd ..
	cd splash
	rm LumenSplash
	cd ..
fi

out="0"
if [ ! -f bin/x86-Release/Lumen ]
then
	cd src
	make
	out="$?"
	cd ..
	if [ "$out" != "0" ]
	then
	    exit
    fi
	cd splash
	make
	cd ..
fi

cnt=0

runLumen() {
    clear
    echo "Password is lumen :)"
	sudo chmod +r /dev/hidraw2
	if [ "$?" == "0" ]
	then
	    ./Lumen #2> /dev/null
	    if [ "$?" != "1" ]
	    then
            let "cnt+=1"
            echo cnt
            if [ $cnt -gt 5 ]
            then
                cnt=0
	            openbox --restart
	            #clear
	            echo "If you can read this, and it's been here a while, please restart the computer."
	            echo " press ctrl-alt-f1, write lumen enter lumen then ctrl+alt+del :)"
	            sleep 2
            fi
	        runLumen
	    fi
	else
	    runLumen
	fi
}

while :
do
    # Run lumen
	cd bin/x86-Release
	runLumen
	cd ../..
    
    # Splash screen
    cd splash
    ./LumenSplash
    if [ "$?" != "0" ]
    then
       exit
    fi
    cd ..
    clear
done

