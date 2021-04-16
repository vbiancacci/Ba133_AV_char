import numpy as np

def geometry():
    fNplus=[]
    fBOre=[]

    #points of detectors without malformations
    pointsN=([grooveOuterRadius,height],
             [radius,height],
             [radius,0.],
             [boreRadius,0.])
    
    pointsB=([boreRadius,0.],
             [boreRadius,boreDepth],
             [0.,boreDepth])   

    #add points for each malformation type
    
    #topgroove
    if any(value!=0 for value in geom['topgroove'].value()):
        topgrooveHeight= geom['topgroove']['depth_in_mm']
        topgrooveRadius= geom['topgroove']['radius_in_mm']
        for n,i in enumerate(pointsN):
            if i==[boreRadius,0.]:
                pointsN[n]=[topgrooveRadius,0.]
                pointsN.insert(n+1,[topgrooveRadius,topgrooveHeight])
                pointsN.insert(n+2,[boreRadius,topgrooveHeight])
        for n,i in enumerate(pointsB):
            if i==[boreRadius,0.]:
                pointsB[n]=[boreRadius,topgrooveHeight]
    #bottom_cyl
    elif any(value!=0 for value in geom['bottom_cyl'].value()):
        bottomcylHeight= geom['bottom_cyl']['depth_in_mm']
        bottomcylTrantisitonHeight= bottomcylHeight - geom['bottom_cyl']['transition_in_mm']
        bottomcylRadius= geom['bottom_cyl']['radius_in_mm']
        for n,i in enumerate(pointsN):
            if i==[radius,height]:
                pointsN[n]=[bottomcylRadius,height]
                pointsN.insert(n+1,[bottomcylRadius,bottomcylHeight])
                pointsN.insert(n+2,[radius,bottomcylTransitionHeight])
    #taper inner
    elif any(value!=0 for value in geom['taper']['top']['inner']):
        taperinnerHeight= geom['taper']['top']['inner']['depth_in_mm']
        taperinnerRadius= math.tan(math.radians(geom['taper']['top']['inner']['angle_in_deg']))*taperinnerHeight + geom['well']['radius_in_mm']
        for n,i in enumerate(pointsN):
            if i==[boreRadius,0.]:
                pointsN[n]=[taperinnerRadius,0.]
                pointsN.insert(n+1,[boreRadius,taperinnerHeight])
        for n,i in enumerate(pointsB):
            if i==[boreRadius,0.]:
                pointsB[n]=[boreRadius,taperinnerHeight]
    #taper outer
    elif any(value!=0 for value in geom['taper']['top']['outer']):
        taperouterHeight= geom['taper']['top']['outer']['height_in_mm']
        taperouterRadius= radius-math.tan(math.radians(geom['taper']['top']['outer']['angle_in_deg']))*taperouterHeight
        for n,i in enumerate(pointsN):
            if i==[radius,0.]:
                pointsN[n]=[radius,taperouterHeight]
                pointsN.insert(n+1,[taperouterRadius,0.])

    #creatation of fNplus and fBore from the TwoDLine between all the points
    for i in range(len(pointsN)-1):
        fNplus.append([TwoDLine(np.array(pointsN[i]),np.array(pointsN[i+1]))])
    fNplus=np.array(fNplus)
   
    for i in range(len(pointsB)-1):
        fBore.append([TwoDLine(np.array(pointsB[i]),np.array(pointsB[i+1]))])
    fBore=np.array(fNplus)
             
    
    return fNplus, fBore
