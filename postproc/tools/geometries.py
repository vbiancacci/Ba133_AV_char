import numpy as np

def geometry():
    #topgroove
    if not all(value ==0 for value in geom['topgroove'].value()):
        topgrooveHeight= geom['topgroove']['depth_in_mm']
        topgrooveRadius= geom['topgroove']['radius_in_mm']
        fNplus=np.array([
            [TwoDLine(np.array([grooveOuterRadius,height]),np.array([radius,height]))], #bottom
            [TwoDLine(np.array([radius,height]),np.array([radius,0.]))], #side
            [TwoDLine(np.array([radius,0.]),np.array([topgrooveRadius,0.]))], #top
            [TwoDLine(np.array([topgrooveRadius,0.]),np.array([topgrooveRadius,topgrooveHeight]))], #top - groove
            [TwoDLine(np.array([topgrooveRadius,topgrooveHeight]),np.array([boreRadius,topgrooveHeight]))], #top - groove
            ])
        fBore=np.array([
            [TwoDLine(np.array([boreRadius,topgrooveHeight]),np.array([boreRadius,boreDepth]))], #top bore hole
            [TwoDLine(np.array([boreRadius,boreDepth]),np.array([0.,boreDepth]))], #top bore hole
            ])
    #bottom cylinder
    elif not all(value ==0 for value in geom['bottom_cyl'].value()):
        bottomcylHeight= geom['bottom_cyl']['depth_in_mm']
        bottomcylTrantisitonHeight= bottomcylHeight - geom['bottom_cyl']['transition_in_mm']
        bottomcylRadius= geom['bottom_cyl']['radius_in_mm']
        fNplus=np.array([
            [TwoDLine(np.array([grooveOuterRadius,height]),np.array([bottomcylRadius,height]))], #bottom
            [TwoDLine(np.array([bottomcylRadius,height]),np.array([bottomcylRadius,bottomcylHeight]))], #bottom cyl
            [TwoDLine(np.array([bottomcylRadius,bottomcylHeight]),np.array([radius,bottomcylTransitionHeight]))], #bottom cyl
            [TwoDLine(np.array([radius,bottomcylTranstionHeight]),np.array([radius,0.]))], #side
            [TwoDLine(np.array([radius,0.]),np.array([boreRadius,0.]))], #top
            ])
        fBore=np.array([
            [TwoDLine(np.array([boreRadius,topgrooveHeight]),np.array([boreRadius,boreDepth]))], #top bore hole
            [TwoDLine(np.array([boreRadius,boreDepth]),np.array([0.,boreDepth]))], #top bore hole
            ])
    #taper inner
    elif not all(value ==0 for value in geom['taper']['top']['inner']):
        taperinnerHeight= geom['taper']['top']['inner']['depth_in_mm']
        taperinnerRadius= math.tan(math.radians(geom['taper']['top']['inner']['angle_in_deg']))*taperinnerHeight + geom['well']['radius_in_mm']
        fNplus=np.array([
            [TwoDLine(np.array([grooveOuterRadius,height]),np.array([radius,height]))], #bottom
            [TwoDLine(np.array([radius,height]),np.array([radius,0.]))], #side
            [TwoDLine(np.array([radius,0.]),np.array([taperinnerRadius,0.]))], #taper
            [TwoDLine(np.array([taperinnerRadius,0.]),np.array([boreRadius,taperinnerHeight]))], #top
            ])
        fBore=np.array([
            [TwoDLine(np.array([boreRadius,0.]),np.array([boreRadius,boreDepth]))], #top bore hole
            [TwoDLine(np.array([boreRadius,boreDepth]),np.array([0.,boreDepth]))], #top bore hole
            ])
    #taper outer     
    elif not all(value ==0 for value in geom['taper']['top']['outer']):
        taperouterHeight= geom['taper']['top']['outer']['height_in_mm']
        taperouterRadius= radius-math.tan(math.radians(geom['taper']['top']['outer']['angle_in_deg']))*taperouterHeight
        fNplus=np.array([
            [TwoDLine(np.array([grooveOuterRadius,height]),np.array([radius,height]))], #bottom
            [TwoDLine(np.array([radius,height]),np.array([radius,taperouterHeight]))], #side
            [TwoDLine(np.array([radius,taperouterHeight]),np.array([taperouterRadius,0.]))], #taper
            [TwoDLine(np.array([taperouterRadius,0.]),np.array([boreRadius,0.]))], #top
            ])
        fBore=np.array([
            [TwoDLine(np.array([boreRadius,0.]),np.array([boreRadius,boreDepth]))], #top bore hole
            [TwoDLine(np.array([boreRadius,boreDepth]),np.array([0.,boreDepth]))], #top bore hole
            ])
    #without malformation
    else:
        fNplus=np.array([
            [TwoDLine(np.array([grooveOuterRadius,height]),np.array([radius,height]))], #bottom
            [TwoDLine(np.array([radius,height]),np.array([radius,0.]))], #side
            [TwoDLine(np.array([radius,0.]),np.array([boreRadius,0.]))], #top
            ])
        fBore=np.array([
            [TwoDLine(np.array([boreRadius,0.]),np.array([boreRadius,boreDepth]))], #top bore hole
            [TwoDLine(np.array([boreRadius,boreDepth]),np.array([0.,boreDepth]))], #top bore hole
            ])
    
    return fNplus, fBore
