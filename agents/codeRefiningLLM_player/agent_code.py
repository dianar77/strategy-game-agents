from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, Action, ActionType
from catanatron.state_functions import (
    get_player_buildings,
    player_key,
    get_player_freqdeck,
    get_longest_road_length,
)

def get_prompt_enhancement():
    """
    Returns strategic advice based on analysis of the game state.
    This advisor helps the agent make optimal decisions in Settlers of Catan.
    """
    # This function will be called before generating a prompt for the LLM
    advice = """# Catan Strategic Advisor

I'll provide you with expert Catan strategy analysis to help you make the best decision. 
Focus on these key strategic principles:

## Probability-Based Resource Collection
- Prioritize high-probability numbers (6, 8, 5, 9) when building settlements
- Balance your resource production to avoid dependency on a single resource
- Pay attention to what resources are scarce in the game

## Resource Value Hierarchy
- Early game: Prioritize brick and wood for initial expansion
- Mid game: Focus on wheat and ore for city upgrades and development cards
- Late game: Maintain balanced resource collection with emphasis on ore/wheat

## Development Card Strategy
- Buy development cards when you have excess sheep/wheat/ore
- Hold knight cards until strategically beneficial (robber placement, largest army)
- Remember that unplayed VP cards count towards your score

## Building Strategy
- Build settlements on resource-diverse intersections
- Position roads to block opponents from prime settlement locations
- Upgrade to cities on tiles with high-probability wheat and ore

## Victory Point Paths
- Cities Path: Ore and wheat focus, upgrade settlements quickly
- Longest Road Path: Wood and brick focus, build a network of roads
- Development Card Path: Sheep, wheat, and ore focus, buy development cards
- Balanced Path: Mix strategies based on your initial settlement positions

## Trading Strategy
- Trade from strength (your abundant resources) to address weaknesses
- Avoid trading resources that directly help opponents reach their next build
- Use maritime trade (ports) when player trading is unfavorable
- Consider the game state when deciding whether to trade

Remember to adapt your strategy based on the current game state and opponents' positions.
"""

    return advice

def analyze_game_state(state, my_color):
    """
    Analyzes the game state to provide specific strategic advice.
    This function isn't called directly by codeRefiningLLM_player but could be added
    to get_prompt_enhancement() in a future version.
    
    Args:
        state: The current game state
        my_color: The color of the player receiving advice
        
    Returns:
        str: Specific strategic advice for the current game state
    """
    # Analyze resource production
    my_resources = get_player_freqdeck(state, my_color)
    
    # Get player's key for accessing state information
    key = player_key(state, my_color)
    
    # Analyze buildings
    settlements = get_player_buildings(state, my_color, SETTLEMENT)
    cities = get_player_buildings(state, my_color, CITY)
    
    # Analyze road length
    road_length = get_longest_road_length(state, my_color)
    
    # Calculate resource inventory
    resource_inventory = {r: my_resources[i] for i, r in enumerate(RESOURCES)}
    
    # Calculate victory points (public information)
    vp = state.player_state.get(f"{key}_VICTORY_POINTS", 0)
    
    # Check development cards
    dev_cards = {}
    for card_type in ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]:
        card_key = f"{key}_{card_type}_IN_HAND"
        count = state.player_state.get(card_key, 0)
        if count > 0:
            dev_cards[card_type] = count
    
    # Determine what resources are most needed
    missing_resources = []
    if resource_inventory.get("BRICK", 0) < 1 and resource_inventory.get("WOOD", 0) < 1:
        missing_resources.append("BRICK/WOOD for roads")
    if resource_inventory.get("WHEAT", 0) < 1:
        missing_resources.append("WHEAT for expansions")
    if resource_inventory.get("ORE", 0) < 2:
        missing_resources.append("ORE for cities/cards")
    
    # Generate strategic advice based on analysis
    advice = []
    
    # Early game advice (0-4 VP)
    if vp < 4:
        advice.append("Focus on expansion: Build roads and settlements to secure resource diversity.")
        if len(settlements) < 3:
            advice.append("Prioritize new settlements to increase resource production.")
        if road_length < 3:
            advice.append("Extend your road network to reach valuable building spots.")
    
    # Mid game advice (4-7 VP)
    elif vp < 7:
        advice.append("Begin transitioning to city upgrades and development cards.")
        if len(cities) < 1:
            advice.append("Upgrade settlements to cities on high-producing wheat/ore spots.")
        if "KNIGHT" in dev_cards:
            advice.append("Consider playing Knight cards strategically to disrupt opponents and work toward Largest Army.")
    
    # Late game advice (7+ VP)
    else:
        advice.append("Execute your final victory plan - analyze exact paths to 10 VP.")
        advice.append("Be selective with trades - only trade if it directly helps your victory plan.")
        if resource_inventory.get("ORE", 0) >= 3 and resource_inventory.get("WHEAT", 0) >= 2:
            advice.append("You have resources for a city - this should be your priority unless you're one VP from victory.")
    
    # Resource-specific advice
    if resource_inventory.get("ORE", 0) >= 3 and resource_inventory.get("WHEAT", 0) >= 2 and len(settlements) > 0:
        advice.append("OPPORTUNITY: You can upgrade a settlement to a city.")
    
    if resource_inventory.get("WOOD", 0) >= 1 and resource_inventory.get("BRICK", 0) >= 1:
        advice.append("OPPORTUNITY: You can build a road to expand your territory.")
    
    if resource_inventory.get("WHEAT", 0) >= 1 and resource_inventory.get("SHEEP", 0) >= 1 and resource_inventory.get("BRICK", 0) >= 1 and resource_inventory.get("WOOD", 0) >= 1:
        advice.append("OPPORTUNITY: You can build a new settlement if you have a valid location.")
    
    if resource_inventory.get("WHEAT", 0) >= 1 and resource_inventory.get("SHEEP", 0) >= 1 and resource_inventory.get("ORE", 0) >= 1:
        advice.append("OPPORTUNITY: Consider buying a development card for tactical advantage.")
    
    # Development card advice
    if "KNIGHT" in dev_cards and dev_cards["KNIGHT"] > 0:
        advice.append("You have Knight card(s) - use them to move the robber to a high-value tile or to block opponents' key resources.")
    
    if "YEAR_OF_PLENTY" in dev_cards:
        advice.append("Use Year of Plenty to acquire the specific resources you need most for your next build.")
    
    if "ROAD_BUILDING" in dev_cards:
        advice.append("Road Building card can help you extend your network or secure the Longest Road bonus.")
    
    if "MONOPOLY" in dev_cards:
        advice.append("Save Monopoly for when players likely have accumulated a specific resource you need.")
    
    # Missing resources advice
    if missing_resources:
        advice.append(f"You're currently missing: {', '.join(missing_resources)}. Consider trading or focusing on these.")
    
    return "\n".join(advice)

def evaluate_settlement_locations(board, player_color):
    """
    Evaluates potential settlement locations based on resource diversity and probability.
    
    Args:
        board: The game board
        player_color: The color of the player
        
    Returns:
        list: Sorted list of (node_id, score) tuples representing settlement location quality
    """
    # This would analyze available settlement locations and score them
    # Implementation would require more complex board analysis
    pass

def calculate_resource_production(board, player_color):
    """
    Calculates expected resource production per roll for a player.
    
    Args:
        board: The game board
        player_color: The color of the player
        
    Returns:
        dict: Expected production rates for each resource
    """
    # This would calculate probability-based production rates
    # Implementation would require detailed board and probability analysis
    pass

def recommend_trade_strategy(state, player_color):
    """
    Recommends a trading strategy based on resource needs and availability.
    
    Args:
        state: The current game state
        player_color: The color of the player
        
    Returns:
        str: Trading strategy recommendation
    """
    # This would analyze current resources and needs to recommend trades
    # Implementation would require more complex resource analysis
    pass

def analyze_winning_path(state, player_color):
    """
    Analyzes possible paths to victory and recommends the most efficient one.
    
    Args:
        state: The current game state
        player_color: The color of the player
        
    Returns:
        str: Recommended path to victory
    """
    # This would analyze the game state and suggest the fastest path to 10 VP
    # Implementation would require complex state analysis and pathfinding
    pass