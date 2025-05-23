def get_prompt_enhancement():
    """
    Returns a string to be added to the beginning of the LLM prompt.
    Analyzes the game state and provides strategic advice tailored to victory.
    """
    
    def provide_advice_based_on_game_state(state):
        """
        Analyzes the state of the game and decides strategic advice.

        Args:
            state: Current game state.

        Returns:
            str: Strategic advice based on game state.
        """
        advice = []

        # Focus on settlement positioning
        advice.append("Focus on placing settlements adjacent to high-probability resource tiles (6, 8, 9) to maximize resource gain.")
        advice.append("Diversify your resource collection to avoid dependency on limited resource types.")
        
        # Development card strategy
        advice.append("Buy development cards early when possible to maximize versatility and increase chances for Largest Army bonus.")
        advice.append("Play Knight cards strategically to disrupt opponents and secure Largest Army.")

        # Robber placement
        advice.append("Place the robber on tiles producing resources critical to opponents' strategies, targeting leading players or highest production tiles.")

        # Resource balancing and trades
        advice.append("Trade cautiously, prioritizing harbors over player trading unless unavoidable. Avoid revealing resource weaknesses.")

        # Longest road strategy
        advice.append("If strategically viable, aim for the Longest Road bonus, connecting settlements to block opponents' expansions.")

        # Consolidate advice
        return "\n".join(advice)

    # Return dynamic game advice preface
    return provide_advice_based_on_game_state