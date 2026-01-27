source busenv/bin/activate
cd /home/pi/mbta
git pull
cp ./bash_profile.sh ~/.bash_profile
if [[ "$HOSTNAME" == "zero" ]]; then
    python3 /home/pi/mbta/mbta_93.py &
else
    python3 /home/pi/mbta/mbta_gl.py &
fi
pi(){
  python3 mbta_93.py
}
