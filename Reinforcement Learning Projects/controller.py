import pystk


def control(aim_point, current_vel):
    
    action = pystk.Action()

    
    if(aim_point[0] > 0):
        action.steer = aim_point[0]*5
    if(aim_point[0] < 0):
        action.steer = aim_point[0]*5
    if(aim_point[0] > .50 or aim_point[0] < -.50):
        action.drift=True
        if (aim_point[0] >.5):
            action.steer=aim_point[0] *5
            
            
        if(aim_point[0] < .5):
            action.steer = aim_point[0] *5

            
            
    else:
        action.drift=False
    if(aim_point[0] <.5 and aim_point[0] > -.5):
        action.nitro = True
        action.acceleration =1
    
    print(aim_point)
    action.acceleration =1

    

    return action



if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)

