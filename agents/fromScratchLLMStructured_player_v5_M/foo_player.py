from catanatron.models.player import Player, Color
from catanatron.models.actions import Action
from catanatron.models.enums import ActionType
# Import building types as constants, not as an enum
from catanatron.models.enums import SETTLEMENT, CITY, ROAD
from catanatron.state_functions import (
    get_player_freqdeck,
    player_can_afford_dev_card,
    get_player_buildings,
    get_actual_victory_points,
    get_visible_victory_points,
    get_longest_road_length,
)
from catanatron import state_functions
from agents.fromScratchLLMStructured_player_v4.llm_tools import LLM

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        self.llm = LLM()  # use self.llm.query_llm(str prompt) to query the LLM
        self.turn_count = 0
        self.is_initial_placement_phase = True

    def decide(self, game, playable_actions):
        self.turn_count += 1
        color = self.color  # get our player color
        state = game.state

        # Get current game state information
        our_points = get_actual_victory_points(state, color)
        our_resources = get_player_freqdeck(state, color)
        our_settlements = get_player_buildings(state, color, SETTLEMENT)
        our_cities = get_player_buildings(state, color, CITY)
        our_roads = get_player_buildings(state, color, ROAD)
        road_length = get_longest_road_length(state, color)
        
        # Determine the current phase of the game more accurately
        # Instead of checking settlement count, check action types
        initial_placement_actions = [
            ActionType.BUILD_SETTLEMENT,
            ActionType.BUILD_ROAD,
        ]
        
        # Check if we're in initial placement based on game state or actions
        if self.is_initial_placement_phase:
            # Check if any action is NOT a placement action
            for action in playable_actions:
                if action.action_type == ActionType.ROLL:
                    self.is_initial_placement_phase = False
                    break
        
        # Handle initial placement phase with better strategy
        if self.is_initial_placement_phase:
            print(f"Initial placement phase - turn {self.turn_count}")
            
            # For initial settlements, use the LLM to evaluate options if multiple exist
            if len(playable_actions) > 1 and any(a.action_type == ActionType.BUILD_SETTLEMENT for a in playable_actions):
                settlement_actions = [(i, a) for i, a in enumerate(playable_actions) if a.action_type == ActionType.BUILD_SETTLEMENT]
                
                if settlement_actions:
                    prompt = f"""
                    I'm placing my initial settlement in Catan.
                    The available settlement locations are:
                    {[f"{i}: {action.value}" for i, action in settlement_actions]}
                    
                    Which location (by index) would be most strategic for resource generation?
                    Consider proximity to high-probability numbers and variety of resources.
                    Respond with just the index number.
                    """
                    response = self.llm.query_llm(prompt)
                    try:
                        if response and response.strip().isdigit():
                            index = int(response.strip())
                            if 0 <= index < len(settlement_actions):
                                return playable_actions[settlement_actions[index][0]]
                    except:
                        pass
            
            # For initial roads, place them in a way that allows for expansion
            if len(playable_actions) > 1 and any(a.action_type == ActionType.BUILD_ROAD for a in playable_actions):
                # Just take the first road option for now
                for action in playable_actions:
                    if action.action_type == ActionType.BUILD_ROAD:
                        return action
            
            # If we reach here, just take the first action
            return playable_actions[0]
        
        # Categorize available actions
        building_actions = []
        dev_card_actions = []
        trade_actions = []
        robber_actions = []
        roll_actions = []
        end_turn_actions = []
        discard_actions = []
        other_actions = []
        
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                building_actions.append(("BUILD_SETTLEMENT", action))
            elif action.action_type == ActionType.BUILD_CITY:
                building_actions.append(("BUILD_CITY", action))
            elif action.action_type == ActionType.BUILD_ROAD:
                building_actions.append(("BUILD_ROAD", action))
            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                dev_card_actions.append(("BUY_DEV_CARD", action))
            elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
                dev_card_actions.append(("PLAY_KNIGHT", action))
            elif action.action_type == ActionType.PLAY_MONOPOLY:
                dev_card_actions.append(("PLAY_MONOPOLY", action))
            elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
                dev_card_actions.append(("PLAY_YEAR_OF_PLENTY", action))
            elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
                dev_card_actions.append(("PLAY_ROAD_BUILDING", action))
            elif action.action_type == ActionType.MARITIME_TRADE:
                trade_actions.append(("MARITIME_TRADE", action))
            elif action.action_type == ActionType.MOVE_ROBBER:
                robber_actions.append(("MOVE_ROBBER", action))
            elif action.action_type == ActionType.ROLL:
                roll_actions.append(("ROLL", action))
            elif action.action_type == ActionType.END_TURN:
                end_turn_actions.append(("END_TURN", action))
            elif action.action_type == ActionType.DISCARD:
                discard_actions.append(("DISCARD", action))
            else:
                other_actions.append((str(action.action_type), action))
        
        # Handle discard actions - keep resources for building
        if discard_actions:
            print(f"Discarding cards - turn {self.turn_count}")
            # Prioritize keeping resources needed for settlements and cities
            return discard_actions[0][1]  # Simple strategy for now: take first discard action
            
        # If we can roll, always roll first
        if roll_actions:
            print(f"Rolling dice - turn {self.turn_count}")
            return roll_actions[0][1]
        
        # If we only have one action, take it (no need to use LLM)
        if len(playable_actions) == 1:
            print(f"Only one action available - turn {self.turn_count}")
            return playable_actions[0]
        
        # If we need to move the robber, use the LLM to decide
        if robber_actions:
            prompt = f"""
            I need to move the robber in a game of Catan. 
            My player has {our_points} points.
            My resources are: {our_resources}.
            I have {len(our_settlements)} settlements and {len(our_cities)} cities.
            
            The available robber moves are:
            {[f"{i}: {action[1].value}" for i, action in enumerate(robber_actions)]}
            
            Which move (by index) would be the most strategic to hurt my opponents while minimizing risk to myself?
            Respond with just the index number.
            """
            response = self.llm.query_llm(prompt)
            try:
                # Try to parse an index from the LLM response
                if response and response.strip().isdigit():
                    index = int(response.strip())
                    if 0 <= index < len(robber_actions):
                        print(f"LLM chose robber action {index} - turn {self.turn_count}")
                        return robber_actions[index][1]
                # If parsing fails, just take the first robber action
                print(f"Using first robber action - turn {self.turn_count}")
                return robber_actions[0][1]
            except:
                print(f"Error parsing LLM response for robber, using first action - turn {self.turn_count}")
                return robber_actions[0][1]
        
        # Enhanced strategic decision making for building and dev cards
        if building_actions or dev_card_actions:
            action_descriptions = []
            combined_actions = []
            
            for i, (desc, action) in enumerate(building_actions + dev_card_actions):
                action_descriptions.append(f"{i}: {desc} {action.value}")
                combined_actions.append(action)
            
            # Enhanced prompt to give the LLM more strategic context
            prompt = f"""
            I'm playing Catan and need to decide my next action.
            My player has {our_points} points out of 10 needed to win.
            My resources are: {our_resources}.
            I have {len(our_settlements)} settlements, {len(our_cities)} cities, and {len(our_roads)} roads.
            My longest road is {road_length} segments long.
            
            The available actions are:
            {action_descriptions}
            
            Which action (by index) would be most strategic for winning the game?
            Consider:
            1. Immediate point gains (cities = 2 VP, settlements = 1 VP)
            2. Resource generation potential
            3. Longest road potential (5 segments = 2 VP)
            4. Development cards that might give Victory Points
            
            Respond with just the index number.
            """
            response = self.llm.query_llm(prompt)
            try:
                # Try to parse an index from the LLM response
                if response and response.strip().isdigit():
                    index = int(response.strip())
                    if 0 <= index < len(combined_actions):
                        print(f"LLM chose building/dev action {index} - turn {self.turn_count}")
                        return combined_actions[index]
            except:
                print(f"Error parsing LLM response for building/dev, using heuristic - turn {self.turn_count}")
            
            # Improved heuristic priorities if LLM parsing fails
            # 1. Build city (2 VP)
            # 2. Build settlement (1 VP)
            # 3. Buy development card (potential VP or Knight)
            # 4. Build road if we're close to longest road (4+ segments)
            # 5. Play development cards
            # 6. Build road otherwise
            
            # Check if we're close to longest road
            close_to_longest_road = road_length >= 4
            
            priority_actions = ["BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEV_CARD"]
            if close_to_longest_road:
                priority_actions.insert(3, "BUILD_ROAD")
                
            for action_type in priority_actions:
                for desc, action in building_actions + dev_card_actions:
                    if desc == action_type:
                        print(f"Using heuristic: {action_type} - turn {self.turn_count}")
                        return action
            
            # Process remaining development card and road actions
            for desc, action in dev_card_actions:
                if desc not in ["BUY_DEV_CARD"]:  # All other dev card plays
                    print(f"Playing development card: {desc} - turn {self.turn_count}")
                    return action
                    
            for desc, action in building_actions:
                if desc == "BUILD_ROAD":
                    print(f"Building road - turn {self.turn_count}")
                    return action
            
            # If we're here, just take the first building or dev card action
            if building_actions:
                print(f"Using first building action - turn {self.turn_count}")
                return building_actions[0][1]
            if dev_card_actions:
                print(f"Using first dev card action - turn {self.turn_count}")
                return dev_card_actions[0][1]
        
        # Improved trade strategy
        if trade_actions:
            # Only trade if we have more than 1 option
            if len(trade_actions) > 1:
                # Create a simple prompt for the LLM to evaluate trades
                trade_descriptions = [f"{i}: {action[1].value}" for i, action in enumerate(trade_actions)]
                prompt = f"""
                I'm trading resources in Catan.
                My current resources are: {our_resources}
                The available trades are:
                {trade_descriptions}
                
                Which trade (by index) would be best given my current resources and buildings?
                Consider which resources I need most for building settlements and cities.
                Respond with just the index number.
                """
                response = self.llm.query_llm(prompt)
                try:
                    if response and response.strip().isdigit():
                        index = int(response.strip())
                        if 0 <= index < len(trade_actions):
                            print(f"LLM chose trade {index} - turn {self.turn_count}")
                            return trade_actions[index][1]
                except:
                    pass
                    
            print(f"Using first trade action - turn {self.turn_count}")
            return trade_actions[0][1]
        
        # If end turn is available and we've done everything else, end the turn
        if end_turn_actions:
            print(f"Ending turn {self.turn_count}")
            return end_turn_actions[0][1]
        
        # If we get here, just take the first action (fallback)
        print(f"Using fallback first action - turn {self.turn_count}")
        return playable_actions[0]