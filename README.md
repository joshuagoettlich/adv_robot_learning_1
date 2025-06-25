ADD YOUR API KEY

docker compose run --rm adv_robot_learning 
 docker exec -it adv_robot_learning-adv_robot_learning-run-a79d3f5d0886 bash

Final_Hand in Launch: 
   catkin_make

    roslaunch om_position_controller position_control.launch sim:=true

    roscd om_position_controller/scripts && \
    python3 spawn_tower_of_hanoi.py 

    roscd om_position_controller/scripts && \
    python3 4_rs_detect_sim.py 

    roscd om_position_controller/scripts && \
    python3 hanoi_state_observer.py

    roscd om_position_controller/scripts && \
    python3 hanoi_llm_robot_logic_node.py 

    roscd om_position_controller/scripts && \
    python3 game_player.py 
    
Solving the puzzle:


<video width="640" height="360" controls>
  <source src="https://raw.githubusercontent.com/joshuagoettlich/adv_robot_learning_1/gasping/ARL_25_noetic_packages/another_solve_fast.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
