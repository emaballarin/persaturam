# SIMULREMO.py
# A just-make-it-work Sanremo 2019 voting simulator

import numpy as np

# NumPy implementation of the "noisy uneven staircase" distribution
def uneven_staircase(elements: int, length: int, uniformity: float, noise_coeff: float):
    # Uniform component
    to_be_returned = np.multiply(np.full(length, 1.0 / length), (elements * (uniformity * (1.0 - noise_coeff))))
    # Noisy component (of the uniform component)
    noise_vector = np.random.rand(length)
    noise_vector = np.multiply(np.true_divide(noise_vector, np.sum(noise_vector)), (elements * (uniformity * noise_coeff)))
    to_be_returned = to_be_returned + noise_vector
    # Non-uniform (i.e. "stairy") component
    remaining_elements: int = elements
    for step in np.arange(length - 1):
        if remaining_elements == 0:
            draw: int = 0
        else:
            draw: int = np.random.randint(remaining_elements + 1)
            remaining_elements = remaining_elements - draw
        to_be_returned[step] = to_be_returned[step] + ((1.0 - uniformity) * draw)
    to_be_returned[length - 1] = to_be_returned[length - 1] + (
        (1.0 - uniformity) * remaining_elements
    )
    # Shuffle and normalize
    np.random.shuffle(to_be_returned)
    return np.true_divide(to_be_returned, elements)


# Do not mindlessly re-use this function. Tweak the internal parameters first!
def castvote(jurysize: int, candidates: int, norm_factor=None):
    if norm_factor is None:
        norm_factor = jurysize
    results = np.multiply((uneven_staircase(jurysize, candidates, 0.6, 0.5)), norm_factor)
    while ((results[(np.argsort(-results)[0])] - results[(np.argsort(-results)[1])]) >= 0.5*(results[(np.argsort(-results)[1])])):
        results = np.multiply((uneven_staircase(jurysize, candidates, 0.6, 0.5)), norm_factor)
    return results


# Automagically manage a chart in the most efficient way
class chart:
    def __init__(self, artists: int):
        self.untidied = np.zeros(artists)
        self.additions: int = 0

    def compound(self, addarray):
        self.untidied = self.untidied + addarray
        self.additions = self.additions + 1


# Parameters
myArtists: int = 24
myJournalists: int = 40
myDemoscopicJurors: int = 300
myHorourableJurors: int = 8
myTelevoters: int = 30000000

simulationsNr: int = 15000

# Track outcomes
alignedNr: int = 0
unalignedNr: int = 0

# Simulate many events
for cnt in np.arange(simulationsNr):

    # Let the event begin!
    mySanremoChart = chart(myArtists)
    myTelevotingChart = chart(myArtists)

    # Prima serata
    demoscopic_votes = castvote(myDemoscopicJurors, myArtists, 0.3)
    press_votes = castvote(myJournalists, myArtists, 0.3)
    tele_votes = castvote(myTelevoters, myArtists, 1)
    mySanremoChart.compound(
        demoscopic_votes + press_votes + np.multiply(tele_votes, 0.4)
    )
    myTelevotingChart.compound(tele_votes)

    # Seconda + Terza serata
    demoscopic_votes = castvote(myDemoscopicJurors, myArtists, 0.3)
    press_votes = castvote(myJournalists, myArtists, 0.3)
    tele_votes = castvote(myTelevoters, myArtists, 1)
    mySanremoChart.compound(
        demoscopic_votes + press_votes + np.multiply(tele_votes, 0.4)
    )
    myTelevotingChart.compound(tele_votes)

    # Quarta serata
    honourable_votes = castvote(myHorourableJurors, myArtists, 0.2)
    press_votes = castvote(myJournalists, myArtists, 0.3)
    tele_votes = castvote(myTelevoters, myArtists, 1)
    mySanremoChart.compound(
        demoscopic_votes + press_votes + np.multiply(tele_votes, 0.5)
    )
    myTelevotingChart.compound(tele_votes)

    # Quinta serata (sfida a 24)
    honourable_votes = castvote(myHorourableJurors, myArtists, 0.2)
    press_votes = castvote(myJournalists, myArtists, 0.3)
    tele_votes = castvote(myTelevoters, myArtists, 1)
    mySanremoChart.compound(
        demoscopic_votes + press_votes + np.multiply(tele_votes, 0.5)
    )
    myTelevotingChart.compound(tele_votes)

    # Quinta serata (sfida a 3)
    final_honourable_votes = castvote(myHorourableJurors, 3, 0.2)
    final_press_votes = castvote(myJournalists, 3, 0.3)
    final_tele_votes = castvote(myTelevoters, 3, 1)

    honourable_votes = np.zeros(myArtists)
    press_votes = np.zeros(myArtists)
    tele_votes_official = np.zeros(myArtists)
    tele_votes_only = np.zeros(myArtists)

    for i in np.arange(3):
        honourable_votes[np.argsort(-mySanremoChart.untidied)][
            i
        ] = final_honourable_votes[i]
        press_votes[np.argsort(-mySanremoChart.untidied)][i] = final_press_votes[i]
        tele_votes_official[np.argsort(-mySanremoChart.untidied)][i] = final_tele_votes[
            i
        ]
        tele_votes_only[np.argsort(-myTelevotingChart.untidied)][i] = final_tele_votes[
            i
        ]

    mySanremoChart.compound(
        demoscopic_votes + press_votes + np.multiply(tele_votes_official, 0.5)
    )
    myTelevotingChart.compound(tele_votes_only)

    SanremoWinnerIndex = np.argsort(-mySanremoChart.untidied)[0]
    SanremoTeleWinnerIndex = np.argsort(-myTelevotingChart.untidied)[0]

    # Update event counters
    if SanremoWinnerIndex == SanremoTeleWinnerIndex:
        alignedNr = alignedNr + 1
    else:
        unalignedNr = unalignedNr + 1

print(unalignedNr / (unalignedNr + alignedNr))
