'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

// Mock data generation
const generateMockData = () => {
  const nodes = [];
  const links = [];
  const prefixes = ['CAM', 'MED', 'IND', 'SENS'];
  
  const anomalousIds = ['SIM-0001', 'MED-0007', 'IND-0003'];
  
  for (let i = 1; i <= 50; i++) {
    let id;
    let isAnomalous = false;
    let trustScore;
    
    if (i <= 3) {
       id = anomalousIds[i - 1];
       isAnomalous = true;
       trustScore = 20 + Math.random() * 15; 
    } else if (i === 4) {
       id = 'CAM-0001'; 
       trustScore = 85;
    } else {
       const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
       id = `${prefix}-${i.toString().padStart(4, '0')}`;
       trustScore = 75 + Math.random() * 20;
    }

    nodes.push({
      id,
      trust_score: trustScore,
      isAnomalous,
      isIsolated: false
    });
  }

  for (let i = 0; i < nodes.length; i++) {
    const numConnections = Math.floor(Math.random() * 3) + 1;
    for (let j = 0; j < numConnections; j++) {
      const targetIdx = Math.floor(Math.random() * nodes.length);
      if (targetIdx !== i) {
         links.push({
           source: nodes[i].id,
           target: nodes[targetIdx].id,
           value: 1
         });
      }
    }
  }

  return { nodes, links };
};

export default function NetworkTopologyMap({ onIsolate }: { onIsolate?: (id: string) => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState(generateMockData());
  const [hoveredNode, setHoveredNode] = useState<any>(null);

  useEffect(() => {
    if (!containerRef.current || !svgRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const colorScale = (score: number, isIsolated: boolean) => {
      if (isIsolated) return '#475569'; // Gray out
      if (score >= 80) return '#22c55e'; // Green
      if (score >= 60) return '#eab308'; // Yellow
      if (score >= 40) return '#f97316'; // Orange
      return '#ef4444'; // Red
    };

    // Filter out links connected to isolated nodes
    const activeLinks = data.links.filter((l: any) => {
       const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
       const targetId = typeof l.target === 'object' ? l.target.id : l.target;
       const sourceNode = data.nodes.find(n => n.id === sourceId);
       const targetNode = data.nodes.find(n => n.id === targetId);
       return !sourceNode?.isIsolated && !targetNode?.isIsolated;
    });

    const simulation = d3.forceSimulation(data.nodes as any)
      .force('link', d3.forceLink(activeLinks).id((d: any) => d.id).distance(70))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius(20));

    const defs = svg.append("defs");
    const filter = defs.append("filter")
        .attr("id", "glow")
        .attr("x", "-50%")
        .attr("y", "-50%")
        .attr("width", "200%")
        .attr("height", "200%");
    
    filter.append("feGaussianBlur")
        .attr("stdDeviation", "8")
        .attr("result", "coloredBlur");
        
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    const link = svg.append('g')
      .attr('stroke', 'rgba(255,255,255,0.2)')
      .attr('stroke-width', 1)
      .selectAll('line')
      .data(activeLinks)
      .join('line');

    const pulseLayer = svg.append('g');
    
    function drag(simulation: any) {
      function dragstarted(event: any) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      }
      function dragged(event: any) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      }
      function dragended(event: any) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }
      return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
    }

    const node = svg.append('g')
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', (d: any) => d.isAnomalous && !d.isIsolated ? 12 : 6)
      .attr('fill', (d: any) => colorScale(d.trust_score, d.isIsolated))
      .attr('stroke', (d: any) => d.isAnomalous && !d.isIsolated ? '#ffffff' : 'none')
      .attr('stroke-width', (d: any) => d.isAnomalous && !d.isIsolated ? 2 : 0)
      .style('cursor', 'pointer')
      .style("filter", (d: any) => d.isAnomalous && !d.isIsolated ? "url(#glow)" : "none")
      .on('mouseover', (event, d: any) => {
         if (d.isAnomalous && !d.isIsolated) setHoveredNode(d);
      })
      .on('mouseout', () => setHoveredNode(null))
      .call(drag(simulation) as any);

    const label = svg.append('g')
      .selectAll('text')
      .data(data.nodes)
      .join('text')
      .attr('dy', 22)
      .attr('text-anchor', 'middle')
      .text((d: any) => d.id)
      .attr('font-size', (d: any) => d.isAnomalous && !d.isIsolated ? '13px' : '9px')
      .attr('font-weight', (d: any) => d.isAnomalous && !d.isIsolated ? 'bold' : 'normal')
      .attr('fill', (d: any) => {
         if (d.isIsolated) return '#475569';
         return d.isAnomalous ? '#ef4444' : 'rgba(255,255,255,0.5)';
      })
      .attr('font-family', 'monospace')
      .style('pointer-events', 'none')
      .style("filter", (d: any) => d.isAnomalous && !d.isIsolated ? "drop-shadow(0px 0px 5px rgba(239,64,64,1))" : "none");

    const containedBadge = svg.append('g')
      .selectAll('text')
      .data(data.nodes.filter(n => n.isIsolated))
      .join('text')
      .attr('dy', -15)
      .attr('text-anchor', 'middle')
      .text('CONTAINED')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('fill', '#22c55e')
      .attr('font-family', 'monospace')
      .style('pointer-events', 'none')
      .style("filter", "drop-shadow(0px 0px 4px rgba(34,197,94,0.8))");

    simulation.on('tick', () => {
      data.nodes.forEach((d: any) => {
        d.x = Math.max(40, Math.min(width - 40, d.x));
        d.y = Math.max(40, Math.min(height - 40, d.y));
      });

      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);
        
      label
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);

      containedBadge
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);

      pulseLayer.selectAll('circle.pulse').remove();
      data.nodes.filter(n => n.isAnomalous && !n.isIsolated).forEach((d: any) => {
         pulseLayer.append('circle')
           .attr('class', 'pulse')
           .attr('cx', d.x)
           .attr('cy', d.y)
           .attr('r', 25)
           .attr('fill', 'none')
           .attr('stroke', '#ef4444')
           .attr('stroke-width', 2)
           .attr('opacity', 0.8)
           .style("filter", "url(#glow)")
           .append("animate")
           .attr("attributeName", "r")
           .attr("values", "12; 40")
           .attr("dur", "1.5s")
           .attr("repeatCount", "indefinite");
           
         pulseLayer.append('circle')
           .attr('class', 'pulse')
           .attr('cx', d.x)
           .attr('cy', d.y)
           .attr('r', 40)
           .attr('fill', 'none')
           .attr('stroke', '#ef4444')
           .attr('stroke-width', 1)
           .attr('opacity', 0.5)
           .append("animate")
           .attr("attributeName", "opacity")
           .attr("values", "0.8; 0")
           .attr("dur", "1.5s")
           .attr("repeatCount", "indefinite");
      });
    });

    return () => {
      simulation.stop();
    };
  }, [data]);

  const handleIsolate = () => {
     if (!hoveredNode) return;
     const nodeId = hoveredNode.id;
     
     // Trigger dashboard event
     if (onIsolate) onIsolate(nodeId);

     setData(prev => {
        const newNodes = prev.nodes.map(n => {
           if (n.id === nodeId) {
              return { ...n, isIsolated: true };
           }
           // Neighbors heal
           if (n.trust_score < 80) {
              return { ...n, trust_score: Math.min(95, n.trust_score + 25) };
           }
           return n;
        });
        return { nodes: newNodes, links: prev.links };
     });
     setHoveredNode(null);
  };

  return (
    <div className="absolute inset-0 w-full h-full" ref={containerRef}>
      <svg ref={svgRef} className="w-full h-full" />
      
      {/* Isolate Button Overlay on hover */}
      {hoveredNode && !hoveredNode.isIsolated && (
         <div 
           className="absolute z-50 pointer-events-auto transition-all"
           style={{ 
             left: hoveredNode.x + 20, 
             top: hoveredNode.y - 20 
           }}
         >
            <button 
               onClick={handleIsolate}
               className="bg-red-600 hover:bg-red-500 text-white text-xs font-bold font-mono px-3 py-1.5 rounded shadow-[0_0_15px_#ef4444] tracking-widest border border-red-400 active:scale-95 transition-all"
               onMouseOver={() => setHoveredNode(hoveredNode)} // keep hover state
            >
               ISOLATE
            </button>
         </div>
      )}
    </div>
  );
}
