'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

export default function NetworkTopologyMap({ 
  onIsolate, 
  onNodeClick, 
  liveScores 
}: { 
  onIsolate?: (id: string) => void; 
  onNodeClick?: (id: string, score: number) => void;
  liveScores?: Record<string, number>;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });
  const [actionNode, setActionNode] = useState<any>(null);
  const simulationRef = useRef<any>(null);
  const linkSelectionRef = useRef<any>(null);
  const containedGroupRef = useRef<any>(null);

  // Initialize network once when we receive initial device list
  useEffect(() => {
    if (data.nodes.length === 0 && liveScores && Object.keys(liveScores).length > 0) {
      const nodes = [];
      const links = [];
      const deviceIds = Object.keys(liveScores);
      
      deviceIds.forEach(id => {
        const score = liveScores[id];
        nodes.push({
          id,
          trust_score: score,
          isAnomalous: score < 60,
          isIsolated: false
        });
      });

      // Generate some connections to make the graph look nice
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
      setData({ nodes, links });
    }
  }, [liveScores, data.nodes.length]);

  // Update node colors/anomalies dynamically without restarting D3 layout
  useEffect(() => {
    if (!svgRef.current || !liveScores || Object.keys(liveScores).length === 0) return;
    
    const svg = d3.select(svgRef.current);
    
    svg.selectAll('circle').each(function(d: any) {
      if (!d) return;
      if (liveScores[d.id] !== undefined) {
        d.trust_score = liveScores[d.id];
        d.isAnomalous = d.trust_score < 60;
      }
      const isAnom = d.isAnomalous && !d.isIsolated;
      // Continuous HSL gradient
      const hue = Math.round((d.trust_score / 100) * 120);
      const sat = d.trust_score < 50 ? 85 : 75;
      const lit = d.trust_score < 30 ? 45 : d.trust_score < 70 ? 50 : 45;
      const isExternal = d.id && String(d.id).includes('.');
      let color = d.isIsolated ? '#475569' : isExternal ? '#0ea5e9' : `hsl(${hue}, ${sat}%, ${lit}%)`;
      
      d3.select(this)
        .attr('fill', color)
        .attr('stroke', isAnom ? '#ffffff' : 'none')
        .attr('stroke-width', isAnom ? 2 : 0)
        .style("filter", isAnom ? "url(#glow)" : "none");
    });
    
    // Update label styles
    svg.selectAll('text').each(function(d: any) {
      if (!d) return;
      // Don't modify the contained badges
      if (d3.select(this).text() === 'CONTAINED') return;
      const isAnom = d.isAnomalous && !d.isIsolated;
      const isExternal = d.id && String(d.id).includes('.');
      d3.select(this)
        .attr('font-size', isAnom ? '13px' : '9px')
        .attr('font-weight', isAnom ? 'bold' : 'normal')
        .attr('fill', d.isIsolated ? '#475569' : isExternal ? '#3edcff' : isAnom ? '#ef4444' : 'rgba(255,255,255,0.5)')
        .style("filter", isAnom ? "drop-shadow(0px 0px 5px rgba(239,64,64,1))" : "none");
    });
  }, [liveScores]);


  useEffect(() => {
    if (!containerRef.current || !svgRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Add click handler to SVG to clear action node
    svg.on('click', () => setActionNode(null));

    // Continuous HSL gradient: red(0) → orange(30) → yellow(50) → green(120)
    const colorScale = (score: number, isIsolated: boolean, id: string) => {
      if (isIsolated) return '#475569';
      if (id && String(id).includes('.')) return '#0ea5e9';
      // Map score 0-100 to hue 0-120 (red to green)
      const hue = Math.round((score / 100) * 120);
      const saturation = score < 50 ? 85 : 75;
      const lightness = score < 30 ? 45 : score < 70 ? 50 : 45;
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
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

    simulationRef.current = simulation;

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

    linkSelectionRef.current = link;

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
      .attr('r', (d: any) => {
         const isExternal = d.id && String(d.id).includes('.');
         if (isExternal) return 4;
         return d.isAnomalous && !d.isIsolated ? 12 : 6;
      })
      .attr('fill', (d: any) => colorScale(d.trust_score, d.isIsolated, d.id))
      .attr('stroke', (d: any) => d.isAnomalous && !d.isIsolated ? '#ffffff' : 'none')
      .attr('stroke-width', (d: any) => d.isAnomalous && !d.isIsolated ? 2 : 0)
      .style('cursor', 'pointer')
      .style("filter", (d: any) => d.isAnomalous && !d.isIsolated ? "url(#glow)" : "none")
      .on('mouseover', function(event, d: any) {
        const isExternal = d.id && String(d.id).includes('.');
        d3.select(this).attr('r', isExternal ? 6 : (d.isAnomalous && !d.isIsolated ? 14 : 8));
      })
      .on('mouseout', function(event, d: any) {
        const isExternal = d.id && String(d.id).includes('.');
        d3.select(this).attr('r', isExternal ? 4 : (d.isAnomalous && !d.isIsolated ? 12 : 6));
      })
      .on('click', (event, d: any) => {
         event.stopPropagation();
         if (d.isAnomalous && !d.isIsolated) {
            setActionNode(d);
         } else {
            setActionNode(null);
         }
         if (onNodeClick) onNodeClick(d.id, d.trust_score);
      })
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
         const isExternal = d.id && String(d.id).includes('.');
         if (isExternal) return '#3edcff';
         return d.isAnomalous ? '#ef4444' : 'rgba(255,255,255,0.5)';
      })
      .attr('font-family', 'monospace')
      .style('pointer-events', 'none')
      .style("filter", (d: any) => d.isAnomalous && !d.isIsolated ? "drop-shadow(0px 0px 5px rgba(239,64,64,1))" : "none");

    containedGroupRef.current = svg.append('g');

    simulation.on('tick', () => {
      const currentWidth = containerRef.current?.clientWidth || width;
      const currentHeight = containerRef.current?.clientHeight || height;
      
      data.nodes.forEach((d: any) => {
        d.x = Math.max(40, Math.min(currentWidth - 40, d.x));
        d.y = Math.max(40, Math.min(currentHeight - 40, d.y));
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

      if (containedGroupRef.current) {
        containedGroupRef.current.selectAll('text')
          .attr('x', (d: any) => d.x)
          .attr('y', (d: any) => d.y);
      }

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
      simulationRef.current = null;
    };
  }, [data]);

  const handleIsolate = () => {
     if (!actionNode) return;
     const nodeId = actionNode.id;
     
     // Trigger dashboard event
     if (onIsolate) onIsolate(nodeId);

     // Mutate the D3 data object directly — no setData, no simulation restart
     const nodeData = data.nodes.find(n => n.id === nodeId);
     if (nodeData) {
       nodeData.isIsolated = true;
     }

     const svg = d3.select(svgRef.current);

     // Gray out the isolated node circle
     svg.selectAll('circle').filter((d: any) => d && d.id === nodeId)
       .attr('fill', '#475569')
       .attr('r', 6)
       .attr('stroke', 'none')
       .attr('stroke-width', 0)
       .style('filter', 'none');

     // Gray out the label
     svg.selectAll('text').filter((d: any) => d && d.id === nodeId)
       .attr('fill', '#475569')
       .attr('font-size', '9px')
       .attr('font-weight', 'normal')
       .style('filter', 'none');

     // Hide links connected to isolated node
     if (linkSelectionRef.current) {
       linkSelectionRef.current.filter((l: any) => {
         const sid = typeof l.source === 'object' ? l.source.id : l.source;
         const tid = typeof l.target === 'object' ? l.target.id : l.target;
         return sid === nodeId || tid === nodeId;
       }).attr('stroke-opacity', 0);
     }

     // Add CONTAINED badge
     if (containedGroupRef.current && nodeData) {
       containedGroupRef.current.append('text')
         .datum(nodeData)
         .attr('x', nodeData.x)
         .attr('y', nodeData.y)
         .attr('dy', -15)
         .attr('text-anchor', 'middle')
         .text('CONTAINED')
         .attr('font-size', '10px')
         .attr('font-weight', 'bold')
         .attr('fill', '#22c55e')
         .attr('font-family', 'monospace')
         .style('pointer-events', 'none')
         .style('filter', 'drop-shadow(0px 0px 4px rgba(34,197,94,0.8))');
     }

     // Remove links from the simulation force so the layout adjusts gently
     if (simulationRef.current) {
       const linkForce = simulationRef.current.force('link') as d3.ForceLink<any, any>;
       if (linkForce) {
         const remainingLinks = linkForce.links().filter((l: any) => {
           const sid = typeof l.source === 'object' ? l.source.id : l.source;
           const tid = typeof l.target === 'object' ? l.target.id : l.target;
           return sid !== nodeId && tid !== nodeId;
         });
         linkForce.links(remainingLinks);
       }
       // Very gentle reheat — just a subtle readjustment, not a full restart
       simulationRef.current.alpha(0.05).restart();
     }

     setActionNode(null);
  };

  return (
    <div className="absolute inset-0 w-full h-full" ref={containerRef}>
      <svg ref={svgRef} className="w-full h-full" />
      
      {/* Isolate Button Overlay on click */}
      {actionNode && !actionNode.isIsolated && (
         <div 
           className="absolute z-50 pointer-events-auto transition-all"
           style={{ 
             left: actionNode.x + 20, 
             top: actionNode.y - 20 
           }}
         >
            <button 
               onClick={handleIsolate}
               className="bg-red-600 hover:bg-red-500 text-white text-xs font-bold font-mono px-3 py-1.5 rounded shadow-[0_0_15px_#ef4444] tracking-widest border border-red-400 active:scale-95 transition-all"
            >
               ISOLATE
            </button>
         </div>
      )}
    </div>
  );
}
