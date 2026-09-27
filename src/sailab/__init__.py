"""SailabAI: a flood digital twin for the Chenab, Ravi and Sutlej rivers around Multan.

Model 1 maps today's flood from Sentinel-1 radar, Model 2 forecasts the flood chance for the next
satellite pass and +1 to +7 days, and the twin loop keeps both up to date as new data arrives.
"""

__version__ = "0.1.0"

# Team rule 6: every output carries this label. False disaster warnings are an offence under the
# NDM Act 2010 (section 35), so nothing we publish may look like an official warning.
DISCLAIMER = "Experimental research output, not an official flood warning."
