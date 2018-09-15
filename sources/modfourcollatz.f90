! Fortran, rev. 1990 source code (Intel IFORT 17.0.1 compliant)
! Copyright (C) 2017, Emanuele Ballarin <emanuele@ballarin.cc>

! MODFOURCOLLATZ
! An efficient modulo-4 optimized Collatz Conjecture (in)validator

PROGRAM MODFOURCOLLATZ

    implicit none

    integer, parameter  :: ik = 8           ! For gfortran-5, 16 is still valid
    integer (kind = ik) :: current, maxchk, cnt, prechain

    integer (kind = ik) :: startfrom = 5    ! Select from where to start

    maxchk = startfrom - 1      ! Numbers before startfrom are assumed to be checked already
    prechain = startfrom

    ! Iterate on every positive integer after startfrom
    do while(.TRUE.)

        cnt = 0
        current = prechain

        ! Unroll the Collatz sequence until an already-checked number is found
        do while(current .GT. maxchk)

            cnt = cnt + 1
            select case (MOD(current, 4))
                case (0)
                    current = current/4
                case (1)
                    ! Optimization subcycle while the sequence is still === 1 MOD4
                    cnt = cnt - 1
                    do while ((MOD(current, 4) == 1) .AND. (current .GT. maxchk))
                        cnt = cnt + 1
                        current = (current - 1)/4
                    end do
                case (2)
                    current = current/2
                case (3)
                    current = (3*current + 1)/2

                case default    ! If this case is selected, there must be a problem
                    print*, 'BEWARE: Computation encountered an error...'
                    print*, 'Checked until ', maxchk
                    print*, 'Error detected while testing ', maxchk + 1, '; Chain-step ', current
                end select
        end do

        maxchk = maxchk + 1
        prechain = prechain + 1

        print*, 'Checked integer: ', maxchk, ' in ', cnt, ' steps'

    end do

END PROGRAM
