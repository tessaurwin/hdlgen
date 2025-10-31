module fsm_lock(
    input clk,
    input rst_n,
    input [3:0] seq,
    output reg unlocked
);

// Define a state machine with four states
reg [3:0] state;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Reset the state to 0 on reset
        state <= 4'b0;
    end else begin
        // Update the state based on the input sequence
        case (state)
            4'b0: begin
                if (seq == 4'b1011) begin
                    state <= 4'b1;
                end
            end
            4'b1: begin
                if (seq == 4'b1011) begin
                    state <= 4'b2;
                end else begin
                    state <= 4'b0;
                end
            end
            4'b2: begin
                if (seq == 4'b1011) begin
                    state <= 4'b3;
                end else begin
                    state <= 4'b0;
                end
            end
            4'b3: begin
                if (seq == 4'b1011) begin
                    unlocked <= 1'b1; // Unlock the FSM
                end else begin
                    state <= 4'b0;
                end
            end
        endcase
    end
end
