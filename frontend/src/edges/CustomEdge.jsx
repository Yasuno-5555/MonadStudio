// Custom Edge with wider click area
import { getBezierPath, BaseEdge } from '@xyflow/react';

export default function CustomEdge({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    markerEnd,
    data,
}) {
    const [edgePath] = getBezierPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
    });

    return (
        <>
            {/* Invisible wider path for easier clicking */}
            <path
                id={`${id}-interaction`}
                d={edgePath}
                fill="none"
                stroke="transparent"
                strokeWidth={20}
                style={{ cursor: 'pointer' }}
            />
            {/* Visible edge */}
            <BaseEdge
                id={id}
                path={edgePath}
                markerEnd={markerEnd}
                style={{ ...style, strokeWidth: 3, stroke: '#58a6ff' }}
            />
        </>
    );
}
