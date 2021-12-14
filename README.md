# kms_fdto

Minimal code to reproduce flip done time out when running with the KMS driver on RPI.

To build & run first time...

```
git clone https://github.com/pjx3/kms_fdto.git
cd kms_fdto
cmake .
make
kms_fdto/kms_fdto
```

Should produce a triangle on screen (you may need to edit the source to set the right render width & height).

To reproduce the flip done timeout

- Edit kms_fdto/kms_fdto.cpp line 1228 to enable the call to snd_pcm_writei
- Rebuild & run
- This time the screen will be black, the time out will show on the kernel log & my code will eventually assert

After this the system is generally unstable, so I usually reboot.

Enable DRM debug output to see where the first timeout occurs (during drm_mode_setcrtc())

```
echo 0x3ff | sudo tee /sys/module/drm/parameters/debug
```
