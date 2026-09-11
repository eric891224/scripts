module leaf;
endmodule

module top;
    if (1) begin : g
        leaf u_leaf();
    end else begin : g
        leaf u_leaf();
    end
endmodule