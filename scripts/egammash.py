#/usr/bin/python3
# -*- coding: ascii -*-

## A Python 3 script to compute the number/type of particles in a gamma/e+/e- electromagnetic shower ##

nElec = 0
nPhot = 1
nPosx = 0
iterations = 1000

oldElec = 0
oldPosx = 0
oldPhot = 0

print(' ')
print('Execution started...')
print(' ')

for i in range(iterations):
    oldElec = nElec
    oldPosx = nPosx
    oldPhot = nPhot
    nElec = oldPhot + oldElec
    nPosx = oldPhot + oldPosx
    nPhot = oldElec + oldPosx

    print(' ')
    print('Photons: ',nPhot)
    print('Electrons: ',nElec)
    print('Positrons: ',nPosx)

    nChg = nElec + nPosx

    print('TOTAL CHARGED: ', nChg)
    print('TOTAL PHOTONS: ', nPhot)

    nTot = nChg + nPhot

    print('TOTAL PARTICLES: ', nTot)
    print(' ')

    UnChRatio = nPhot/nTot

    print('UNCHARGED/TOTAL RATIO: ', UnChRatio)
    print('-------------------------------------------------------------------')

print(' ')
print('Execution successful!')
print(' ')
