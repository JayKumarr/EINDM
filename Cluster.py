import Contant as con
class Cluster:
    def __init__(self, CF):
        self.widFreq = CF[con.I_cwf]  # maintaining wordId and the occurance
        self.widToWidFreq = CF[con.I_cww]
