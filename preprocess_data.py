# Heuristic
from heuristic import lh, lh_oracle

ECB='ecb'
GVC='gvc'
# LH - ECB
lh(ECB, threshold=0.05)

# LH - GVC
lh(GVC, threshold=0.05)

# LH_ORACLE - ECB, GVC
lh_oracle(ECB, threshold=0.05)
lh_oracle(GVC, threshold=0.05)