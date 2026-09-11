module leaf (
    input  logic a,
    output logic y
);
    assign y = a;
endmodule

module middle (
    input  logic a,
    output logic y
);
    leaf u_leaf (
        .a(a),
        .y(y)
    );
endmodule

module top (
    input  logic a,
    output logic y
);
    logic tmp;

    middle u_middle (
        .a(a),
        .y(tmp)
    );

    leaf u_leaf2 (
        .a(tmp),
        .y(y)
    );
endmodule