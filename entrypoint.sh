#!/bin/bash
echo "--- Setting up container, please wait..."
echo "--- Now as root..."

su - root <<!
root
service ssh start
setfacl -R -d -m u::rwx,g::rwx,o::rwx /workspace
setfacl -R -m u::rwx,g::rwx,o::rwx /workspace
!

echo "--- End of root..."
echo "--- Additional installations..."
pip install opencv-python
apt-get install libdynamicedt3d-dev
echo "--- Additional container setup completed, ready for work..."
tail -F /dev/null