import random
from typing import List
import logging

from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from catanatron.state_functions import (
    get_player_buildings, 
    get_actual_victory_points,
    player_num_resource_cards
)
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        self.llm = LLM()  # use self.llm.query_llm(str prompt) to query the LLM
        print(f"Initialized FooPlayer with color {self.color}")
        
    def decide(self, game, playable_actions):
        """
        Decides on the action to take based on the current game state 
        and available actions.
        """
        if not playable_actions:
            return None
            
        print(f"FooPlayer deciding from {len(playable_actions)} actions")
        
        # Rank actions based on their strategic value
        ranked_actions = self.rank_actions(game, playable_actions)
        
        # Choose the highest ranked action
        chosen_action = ranked_actions[0]
        print(f"FooPlayer chose action: {chosen_action.action_type}")
        return chosen_action
    
    def rank_actions(self, game, playable_actions):
        """
        Ranks the available actions based on strategic value.
        Returns list of actions sorted by priority (highest first).
        """
        scored_actions = []
        for action in playable_actions:
            score = self.evaluate_action(game, action)
            scored_actions.append((score, action))
        
        # Sort by score (descending) and return just the actions
        scored_actions.sort(reverse=True)
        return [action for score, action in scored_actions]
    
    def evaluate_action(self, game, action):
        """
        Evaluates an action and returns a score representing its value.
        Higher scores are better.
        """
        score = 0
        
        # Evaluate settlement placement - highest priority for good locations
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            node_id = action.value
            # Get production potential of this node
            node_production_counter = game.state.board.map.node_production[node_id]
            
            # Value based on production probability
            production_value = sum(node_production_counter.values())
            
            # Value resource diversity
            diversity_value = len(node_production_counter) * 2
            
            # Check if near a port
            port_value = 0
            for resource, port_nodes in game.state.board.map.port_nodes.items():
                if node_id in port_nodes:
                    port_value = 5  # Value ports
            
            # Combine scores with appropriate weights
            score = 50 + (production_value * 10) + diversity_value + port_value
            print(f"Settlement at {node_id} scored: {score} (prod:{production_value}, div:{diversity_value}, port:{port_value})")
        
        # Evaluate city building - high priority for resource production
        elif action.action_type == ActionType.BUILD_CITY:
            node_id = action.value
            # Cities double production, so they're very valuable
            score = 45
            # Check what resources this city would produce
            node_production = game.state.board.map.node_production[node_id]
            if node_production:
                # Extra points based on production value
                score += sum(node_production.values()) * 5
        
        # Evaluate road building - moderate priority for expansion
        elif action.action_type == ActionType.BUILD_ROAD:
            score = 30
            # Roads that lead to good settlement spots should be valued higher
            # This is a simple implementation that will be improved later
        
        # Evaluate development card purchase
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            score = 25
            
        # Evaluate robber actions - prioritize blocking opponents
        elif action.action_type == ActionType.MOVE_ROBBER:
            tile_id = action.value[0]
            victim_color = action.value[1]
            
            # Higher score if we're targeting a player with more points
            if victim_color is not None:
                victim_vps = get_actual_victory_points(game.state, victim_color)
                score = 20 + (victim_vps * 2)
                
                # Check if they have resources to steal
                victim_resources = player_num_resource_cards(game.state, victim_color)
                score += min(victim_resources, 5)  # Value up to 5 resources
            else:
                score = 20
        
        # For rolling dice, just do it
        elif action.action_type == ActionType.ROLL:
            score = 100  # Always roll when possible
        
        # For ending turn, low priority but necessary
        elif action.action_type == ActionType.END_TURN:
            score = 10
            
        # For other action types, assign a moderate score
        else:
            score = 15
            
        # Add some randomness to avoid predictable behavior
        score += random.random()
        
        return score