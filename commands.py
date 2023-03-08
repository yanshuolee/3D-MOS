R.from_quat([0,-0.7071067811865475,0,0.7071067811865475]).as_euler("xyz", degrees=True)

R.from_quat([0,0,1,0]).as_euler("xyz", degrees=True)

R.from_euler('xyz', [0, 0, 60], degrees=True).as_quat()

